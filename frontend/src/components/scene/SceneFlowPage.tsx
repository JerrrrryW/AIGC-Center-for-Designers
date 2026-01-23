import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Step,
  StepLabel,
  Stepper,
  TextField,
  Typography,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';
import { Responsive, WidthProvider, type Layout, type Layouts } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api';
import { SectionCard, type LayoutConfig, type LayoutSection, type MediaSlot } from './dynamic/DynamicRenderer';

const ResponsiveGridLayout = WidthProvider(Responsive);

type ChatMessage = { role: 'user' | 'bot'; text: string; form?: FormSection[]; typing?: boolean };
type FormSection = { title: string; options: string[] };

type LayoutPayload = {
  layout_config?: any;
  layoutConfig?: any;
  text_response?: string;
  textResponse?: string;
};

type StatusState = {
  type: 'idle' | 'loading' | 'success' | 'error';
  message: string;
};

const CANVAS_DIMENSION = 4000;
const TARGET_BG_WIDTH = 2400;
const FLOW_STEPS = ['对话澄清', '编辑与素材', '预览入画布'];
const DASHBOARD_MIN_CARD = 280;
const GRID_BREAKPOINTS = { lg: 1200, md: 900, sm: 600, xs: 0 };
const GRID_COLS = { lg: 12, md: 8, sm: 6, xs: 4 };
const GRID_ROW_HEIGHT = 32;
const GRID_MARGIN: [number, number] = [16, 16];

const SceneFlowPage: React.FC = () => {
  const navigate = useNavigate();
  const [status, setStatus] = useState<StatusState>({ type: 'idle', message: '' });
  const [showQuick, setShowQuick] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'bot', text: '你好！告诉我你想生成什么场景，我会先用选择题帮你把需求变清晰。' },
  ]);
  const [input, setInput] = useState('');
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string[]>>({});
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig | undefined>(undefined);
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [isWorking, setIsWorking] = useState(false);
  const [uiState, setUiState] = useState<Record<string, any>>({});
  const [previewItems, setPreviewItems] = useState<any[]>([]);

  const [quickPrompt, setQuickPrompt] = useState('');
  const [quickImage, setQuickImage] = useState<string | null>(null);
  const [quickStatus, setQuickStatus] = useState<StatusState>({ type: 'idle', message: '' });
  const [quickLoading, setQuickLoading] = useState(false);

  const latestForm = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const msg = messages[i];
      if (msg.role === 'bot' && msg.form?.length) return msg.form;
    }
    return null;
  }, [messages]);

  const isFormComplete = useMemo(() => {
    if (!latestForm?.length) return true;
    return latestForm.every((section) => (selectedOptions[section.title] ?? []).length > 0);
  }, [latestForm, selectedOptions]);
  const hasForm = Boolean(latestForm?.length);

  const chatHistory = useMemo(() => {
    return messages
      .filter((m) => !m.typing)
      .map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.text}`)
      .join('\n');
  }, [messages]);

  const aspectRatio = useMemo(() => {
    const ratioField = layoutConfig?.meta?.aspect_ratio_field;
    const fromUi = ratioField ? uiState?.[ratioField] : undefined;
    const fromSelected = selectedOptions['画幅比例']?.[0];
    const fromMeta = layoutConfig?.meta?.aspect_ratio;
    return fromUi || fromSelected || fromMeta || '3:4';
  }, [layoutConfig?.meta?.aspect_ratio, layoutConfig?.meta?.aspect_ratio_field, selectedOptions, uiState]);

  const mediaSlots = useMemo(
    () => extractMediaSlots(layoutConfig, uiState),
    [layoutConfig, uiState],
  );
  const activeStep = previewItems.length ? 2 : layoutConfig ? 1 : 0;

  useEffect(() => {
    if (!layoutConfig) return;
    setUiState((prev) => {
      if (Object.keys(prev || {}).length > 0) {
        return prev;
      }
      return initializeUiState(layoutConfig);
    });
  }, [layoutConfig]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  const toggleOption = useCallback((title: string, option: string) => {
    setSelectedOptions((prev) => {
      const current = prev[title] ?? [];
      const next = current.includes(option)
        ? current.filter((v) => v !== option)
        : [...current, option];
      return { ...prev, [title]: next };
    });
  }, []);

  const handleSend = useCallback(async () => {
    const value = input.trim();
    if (!value || isSending) return;
    setIsSending(true);
    setStatus({ type: 'loading', message: '正在生成对话回复...' });
    setInput('');
    const nextMessages = [...messages, { role: 'user' as const, text: value }];
    setMessages([...nextMessages, { role: 'bot', text: 'Agent 输入中...', typing: true }]);
    queueMicrotask(scrollToBottom);

    try {
      const res = await api.post('/chat', {
        messages: nextMessages.map((m) => ({
          role: m.role === 'bot' ? 'assistant' : 'user',
          content: m.text,
        })),
        selected_options: selectedOptions,
      });
      const obj = res.data?.json_object ?? res.data;
      const botText = obj?.text_response ?? '好的，我们继续。';
      const form = normalizeForm(obj?.form);
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => !m.typing);
        return [...withoutTyping, { role: 'bot', text: botText, form }];
      });
      setStatus({ type: 'success', message: '对话已更新，可继续补充或进入编辑。' });
    } catch (error) {
      let message = '连接服务器失败，请检查后端服务。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data?.error || error.response.data?.message || message;
      }
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => !m.typing);
        return [...withoutTyping, { role: 'bot', text: message }];
      });
      setStatus({ type: 'error', message });
    } finally {
      setIsSending(false);
      queueMicrotask(scrollToBottom);
    }
  }, [input, isSending, messages, scrollToBottom, selectedOptions]);

  const handleGenerateEdit = useCallback(async () => {
    if (!isFormComplete || isSending) return;
    const initialPrompt = messages.find((m) => m.role === 'user')?.text ?? '';
    setStatus({ type: 'loading', message: '正在生成编辑界面...' });
    setIsWorking(true);
    setPreviewItems([]);

    try {
      const [layoutRes] = await Promise.all([
        api.post('/generate-layout', {
          prompt: initialPrompt,
          chat_history: chatHistory,
          selected_options: selectedOptions,
        }),
        api.post('/analyze-layer-plan', {
          chat_history: chatHistory,
          selected_options: selectedOptions,
          ui_state: {},
        }),
      ]);

      const layoutPayload: LayoutPayload = layoutRes.data;
      const nextLayout = layoutPayload.layout_config ?? layoutPayload.layoutConfig;
      setLayoutConfig(nextLayout);
      setGeneratedPrompt(layoutPayload.text_response ?? layoutPayload.textResponse ?? '');
      setUiState(initializeUiState(nextLayout));
      setStatus({ type: 'success', message: '编辑界面已就绪，开始准备素材吧。' });
    } catch (error) {
      let message = '生成编辑界面失败。';
      if (axios.isAxiosError(error) && error.response) {
        message = error.response.data?.error || error.response.data?.message || message;
      }
      setMessages((prev) => [...prev, { role: 'bot', text: message }]);
      setStatus({ type: 'error', message });
    } finally {
      setIsWorking(false);
    }
  }, [chatHistory, isFormComplete, isSending, messages, selectedOptions]);

  const handleStateChange = useCallback((componentId: string, value: any) => {
    setUiState((prev) => ({ ...prev, [componentId]: value }));
  }, []);

  const updateSlot = useCallback(
    (componentId: string, slotId: string, updater: (slot: MediaSlot) => MediaSlot) => {
      setUiState((prev) => {
        const slots = prev[componentId];
        if (!Array.isArray(slots)) return prev;
        const next = slots.map((slot: MediaSlot) =>
          slot.id === slotId ? updater(slot) : slot,
        );
        return { ...prev, [componentId]: next };
      });
    },
    [],
  );

  const handleSlotUpload = useCallback(
    async (componentId: string, slotId: string, file: File) => {
      if (!file.type.startsWith('image/')) return;
      const dataUrl = await fileToDataUrl(file);
      updateSlot(componentId, slotId, (slot) => ({ ...slot, uri: dataUrl }));
    },
    [updateSlot],
  );

  const handleSlotGenerate = useCallback(
    async (componentId: string, slot: MediaSlot) => {
      setIsWorking(true);
      setStatus({ type: 'loading', message: `正在生成素材：${slot.label}` });
      try {
        const isBackground = slot.layerType === 'background';
        const size = resolveSizeForSlot(isBackground, aspectRatio);
        const imgRes = await api.post('/generate-image', {
          prompt: slot.prompt,
          width: size.width,
          height: size.height,
        });
        const base64 = imgRes.data?.image_base64;
        if (!base64) throw new Error('未返回图片数据');
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        updateSlot(componentId, slot.id, (prev) => ({ ...prev, uri: dataUrl }));
        setStatus({ type: 'success', message: '素材生成完成。' });
      } catch (error) {
        const msg = toApiErrorMessage(error, '生成素材失败。');
        setStatus({ type: 'error', message: msg });
        window.alert(msg);
      } finally {
        setIsWorking(false);
      }
    },
    [aspectRatio, updateSlot],
  );

  const handleGenerateAllSlots = useCallback(async () => {
    if (!layoutConfig) {
      setStatus({ type: 'error', message: '请先生成编辑界面。' });
      return;
    }
    const ownerMap = new Map<string, string>();
    (layoutConfig.sections ?? []).forEach((section) => {
      (section.components ?? []).forEach((component) => {
        if (component.type === 'media-uploader') {
          const stateSlots = uiState[component.id];
          if (Array.isArray(stateSlots)) {
            stateSlots.forEach((slot: MediaSlot) => {
              ownerMap.set(slot.id, component.id);
            });
          }
        }
      });
    });

    const slotsToGenerate = mediaSlots.filter((slot) => !slot.uri);
    if (!slotsToGenerate.length) {
      setStatus({ type: 'success', message: '所有素材已就绪，无需再生成。' });
      return;
    }

    setIsWorking(true);
    try {
      for (let i = 0; i < slotsToGenerate.length; i += 1) {
        const slot = slotsToGenerate[i];
        const componentId = ownerMap.get(slot.id);
        if (!componentId) {
          continue;
        }
        setStatus({
          type: 'loading',
          message: `正在生成素材 ${i + 1}/${slotsToGenerate.length}：${slot.label}`,
        });
        const isBackground = slot.layerType === 'background';
        const size = resolveSizeForSlot(isBackground, aspectRatio);
        const imgRes = await api.post('/generate-image', {
          prompt: slot.prompt,
          width: size.width,
          height: size.height,
        });
        const base64 = imgRes.data?.image_base64;
        if (!base64) {
          throw new Error('未返回图片数据');
        }
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        updateSlot(componentId, slot.id, (prev) => ({ ...prev, uri: dataUrl }));
      }
      setStatus({ type: 'success', message: `批量生成完成（${slotsToGenerate.length} 个）。` });
    } catch (error) {
      const msg = toApiErrorMessage(error, '批量生成失败。');
      setStatus({ type: 'error', message: msg });
      window.alert(msg);
    } finally {
      setIsWorking(false);
    }
  }, [aspectRatio, layoutConfig, mediaSlots, uiState, updateSlot]);

  const handleSlotRemoveBackground = useCallback(
    async (componentId: string, slot: MediaSlot) => {
      if (!slot.uri) return;
      setIsWorking(true);
      setStatus({ type: 'loading', message: `正在抠图：${slot.label}` });
      try {
        const base64 = extractBase64(slot.uri);
        const res = await api.post('/remove-background', { image_base64: base64 });
        const out = res.data?.image_base64;
        if (!out) throw new Error('抠图未返回结果');
        const dataUrl = out.startsWith('data:') ? out : `data:image/png;base64,${out}`;
        updateSlot(componentId, slot.id, () => ({
          ...slot,
          uri: dataUrl,
          hasTransparentBg: true,
        }));
        setStatus({ type: 'success', message: '抠图完成。' });
      } catch (error) {
        const msg = toApiErrorMessage(error, '抠图失败。');
        setStatus({ type: 'error', message: msg });
        window.alert(msg);
      } finally {
        setIsWorking(false);
      }
    },
    [updateSlot],
  );

  const handleGeneratePreviewToCanvas = useCallback(async () => {
    setIsWorking(true);
    setStatus({ type: 'loading', message: '正在组装预览...' });
    try {
      const materials = mediaSlots
        .filter((s) => s.layerType !== 'background')
        .map((s) => ({
          name: s.label,
          layer_type: s.layerType,
          image_base64: s.uri ? extractBase64(s.uri) : undefined,
        }));

      const res = await api.post('/compose-materials', {
        aspect_ratio: aspectRatio,
        materials,
      });
      const layers = res.data?.layers || [];
      if (!Array.isArray(layers) || layers.length === 0) {
        throw new Error('组装接口未返回图层');
      }

      const bgSlot = mediaSlots.find((s) => s.layerType === 'background');
      const bgUri = bgSlot?.uri;
      const bgW = 768;
      const bgH = Math.round(768 / parseAspectRatio(aspectRatio));

      const pendingItems: any[] = [];
      {
        const dataUrl =
          bgUri ??
          buildPlaceholderDataUrl(`background:${generatedPrompt || ''}`, bgW, bgH);
        const bgPlacement = mapToCanvas({
          x: 0,
          y: 0,
          width: bgW,
          height: bgH,
          zIndex: 0,
          bgW,
          bgH,
        });
        pendingItems.push({ dataUrl, name: '背景', ...bgPlacement });
      }

      layers.forEach((layer: any) => {
        const dataUrl = layer.image_base64?.startsWith('data:')
          ? layer.image_base64
          : `data:image/png;base64,${layer.image_base64}`;
        const placement = layer.placement || {};
        const mapped = mapToCanvas({
          x: placement.x ?? 0,
          y: placement.y ?? 0,
          width: layer.width ?? 512,
          height: layer.height ?? 512,
          zIndex: placement.z_index ?? 1,
          bgW,
          bgH,
        });
        pendingItems.push({
          dataUrl,
          name: layer.name || layer.id || '图层',
          ...mapped,
        });
      });

      sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
      setPreviewItems(pendingItems);
      setStatus({ type: 'success', message: '预览完成，可进入画布。' });
    } catch (error) {
      const msg = toApiErrorMessage(error, '生成预览失败。');
      setStatus({ type: 'error', message: msg });
      window.alert(msg);
    } finally {
      setIsWorking(false);
    }
  }, [aspectRatio, generatedPrompt, mediaSlots]);

  const handleQuickGenerate = useCallback(async () => {
    const prompt = quickPrompt.trim();
    if (!prompt || quickLoading) return;
    setQuickLoading(true);
    setQuickStatus({ type: 'loading', message: '正在生成快速图像...' });
    try {
      const res = await api.post('/generate-image', {
        prompt,
        width: 768,
        height: 768,
      });
      const base64 = res.data?.image_base64;
      if (!base64) throw new Error('未返回图片数据');
      const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
      setQuickImage(dataUrl);
      setQuickStatus({ type: 'success', message: '快速图像已生成。' });
    } catch (error) {
      const msg = toApiErrorMessage(error, '快速生成失败。');
      setQuickStatus({ type: 'error', message: msg });
    } finally {
      setQuickLoading(false);
    }
  }, [quickLoading, quickPrompt]);

  const handleQuickAddToCanvas = useCallback(() => {
    if (!quickImage) return;
    const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
    let pendingItems: { dataUrl: string; name: string }[] = [];
    if (pendingRaw) {
      try {
        pendingItems = JSON.parse(pendingRaw);
      } catch {
        pendingItems = [];
      }
    }
    pendingItems.push({ dataUrl: quickImage, name: `Scene-${Date.now()}.png` });
    sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
    navigate('/canvas');
  }, [navigate, quickImage]);

  const handleGoCanvas = useCallback(() => {
    if (!previewItems.length) {
      setStatus({ type: 'error', message: '请先生成预览。' });
      return;
    }
    navigate('/canvas');
  }, [navigate, previewItems.length]);

  const previewData = useMemo(() => {
    const bgSlot = mediaSlots.find((slot) => slot.layerType === 'background');
    const subjects = mediaSlots.filter((slot) => slot.layerType !== 'background');
    return { bgSlot, subjects };
  }, [mediaSlots]);

  const layoutMeta = useMemo(() => (layoutConfig?.meta ?? {}).layout ?? {}, [layoutConfig]);
  const layoutColumns = Math.max(1, Math.floor(coerceNumber(layoutMeta.columns) ?? 5));
  const layouts = useMemo<Layouts>(() => {
    const sections = layoutConfig?.sections ?? [];
    const spanToWidth = (cols: number, span: number) => {
      const colUnit = Math.max(1, Math.floor(cols / layoutColumns));
      return Math.min(cols, Math.max(1, span * colUnit));
    };
    const buildForCols = (cols: number) => {
      const items: LayoutItemSpec[] = [];
      const majorSpan = Math.min(2, layoutColumns);
      items.push({ i: 'chat', w: spanToWidth(cols, majorSpan), h: 12 });
      items.push({ i: 'form', w: spanToWidth(cols, 1), h: 7 });
      if (layoutConfig) {
        items.push({ i: 'summary', w: spanToWidth(cols, majorSpan), h: 4 });
        sections.forEach((section) => {
          const span = Math.max(1, Math.floor(coerceNumber(section.layout?.span) ?? 1));
          items.push({
            i: `section-${section.id}`,
            w: spanToWidth(cols, Math.min(span, layoutColumns)),
            h: estimateSectionHeight(section),
          });
        });
        items.push({ i: 'preview', w: spanToWidth(cols, majorSpan), h: 10 });
      }
      if (showQuick) {
        items.push({ i: 'quick', w: spanToWidth(cols, 1), h: 7 });
      }
      return packGridItems(items, cols);
    };
    return {
      lg: buildForCols(GRID_COLS.lg),
      md: buildForCols(GRID_COLS.md),
      sm: buildForCols(GRID_COLS.sm),
      xs: buildForCols(GRID_COLS.xs),
    };
  }, [layoutColumns, layoutConfig, showQuick]);

  const actionDisabled = isSending || isWorking;
  const actionBar = useMemo(() => {
    if (!layoutConfig) {
      return (
        <Stack direction="row" spacing={1} alignItems="center">
          <Button
            variant="contained"
            onClick={handleGenerateEdit}
            disabled={isSending || !isFormComplete}
          >
            生成编辑界面
          </Button>
          <Button variant="text" onClick={() => setShowQuick((prev) => !prev)} disabled={isSending}>
            {showQuick ? '隐藏快速模式' : '快速模式'}
          </Button>
        </Stack>
      );
    }
    return (
      <Stack direction="row" spacing={1} alignItems="center">
        <Button
          variant="contained"
          onClick={handleGeneratePreviewToCanvas}
          disabled={actionDisabled}
        >
          生成预览
        </Button>
        <Button
          variant="outlined"
          onClick={handleGoCanvas}
          disabled={actionDisabled || !previewItems.length}
        >
          进入画布
        </Button>
        <Button variant="text" onClick={() => setShowQuick((prev) => !prev)} disabled={actionDisabled}>
          {showQuick ? '隐藏快速模式' : '快速模式'}
        </Button>
      </Stack>
    );
  }, [
    actionDisabled,
    handleGenerateEdit,
    handleGeneratePreviewToCanvas,
    handleGoCanvas,
    isFormComplete,
    isSending,
    layoutConfig,
    previewItems.length,
    showQuick,
  ]);

  return (
    <Container
      maxWidth={false}
      sx={{
        mt: 3,
        mb: 2,
        height: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        overflow: 'hidden',
        position: 'relative',
        pb: 10,
      }}
    >
      <Typography variant="h4" component="h1" gutterBottom>
        AI 场景生成
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 1 }}>
        {FLOW_STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', pb: 12, pr: 1 }}>
        <ResponsiveGridLayout
          className="scene-dashboard-grid"
          layouts={layouts}
          breakpoints={GRID_BREAKPOINTS}
          cols={GRID_COLS}
          rowHeight={GRID_ROW_HEIGHT}
          margin={GRID_MARGIN}
          containerPadding={[8, 8]}
          isDraggable={false}
          isResizable={false}
          compactType="vertical"
          useCSSTransforms
        >
          <div key="chat">
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)',
              }}
            >
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1, minHeight: 0 }}>
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6">聊天窗</Typography>
                    <Typography variant="caption" color="text.secondary">
                      多轮对话澄清需求
                    </Typography>
                  </Box>
                  <Chip
                    label={hasForm ? (isFormComplete ? '可进入编辑' : '待完成') : '待生成'}
                    size="small"
                  />
                </Stack>

                <Paper
                  ref={scrollRef}
                  variant="outlined"
                  sx={{
                    flex: 1,
                    minHeight: 0,
                    overflowY: 'auto',
                    p: 1.5,
                    backgroundColor: '#f9fafb',
                  }}
                >
                  <Stack spacing={1.2}>
                    {messages.map((msg, idx) => (
                      <Box
                        key={`${msg.role}-${idx}`}
                        sx={{
                          display: 'flex',
                          justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        }}
                      >
                        <Box
                          sx={{
                            px: 1.5,
                            py: 1,
                            maxWidth: '84%',
                            borderRadius: 2,
                            backgroundColor: msg.role === 'user' ? 'primary.main' : 'grey.100',
                            color: msg.role === 'user' ? 'primary.contrastText' : 'text.primary',
                          }}
                        >
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                            {msg.text}
                            {msg.typing ? <LoadingDots /> : null}
                          </Typography>
                        </Box>
                      </Box>
                    ))}
                  </Stack>
                </Paper>

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="请输入你想生成的内容"
                    fullWidth
                    multiline
                    minRows={2}
                    onKeyDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault();
                        handleSend();
                      }
                    }}
                  />
                  <IconButton color="primary" onClick={handleSend} disabled={isSending}>
                    <SendIcon />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          </div>

          <div key="form">
            <Card sx={{ height: '100%', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)' }}>
              <CardContent
                sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%', overflowY: 'auto' }}
              >
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6">表单卡片</Typography>
                    <Typography variant="caption" color="text.secondary">
                      选择题结果决定编辑项
                    </Typography>
                  </Box>
                  <Chip
                    label={hasForm ? (isFormComplete ? '已完成' : '待选择') : '待生成'}
                    size="small"
                  />
                </Stack>

                {latestForm?.length ? (
                  <Stack spacing={1.5}>
                    {latestForm.map((section, sectionIndex) => (
                      <Box key={`${section.title}-${sectionIndex}`}>
                        <Typography variant="caption" sx={{ opacity: 0.9, display: 'block' }}>
                          {section.title}
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                          {(Array.isArray(section.options) ? section.options : []).map((opt) => {
                            const selected = selectedOptions[section.title]?.includes(opt);
                            return (
                              <Chip
                                key={opt}
                                label={opt}
                                size="small"
                                clickable
                                color={selected ? 'primary' : 'default'}
                                variant={selected ? 'filled' : 'outlined'}
                                onClick={() => toggleOption(section.title, opt)}
                              />
                            );
                          })}
                        </Stack>
                      </Box>
                    ))}
                    {!isFormComplete ? (
                      <Typography variant="caption" color="error">
                        请为每项至少选择一个选项，才能进入编辑界面。
                      </Typography>
                    ) : null}
                  </Stack>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    暂无表单，请先在聊天中描述需求。
                  </Typography>
                )}
              </CardContent>
            </Card>
          </div>

          {layoutConfig ? (
            <div key="summary">
              <Card sx={{ height: '100%', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)' }}>
                <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography variant="h6">编辑目标概要</Typography>
                      <Typography variant="caption" color="text.secondary">
                        当前配置摘要
                      </Typography>
                    </Box>
                    <Chip label="编辑阶段" size="small" />
                  </Stack>
                  <Typography variant="body2" color="text.secondary">
                    {layoutConfig.meta?.summary || '在下方配置素材，完成后生成预览。'}
                  </Typography>
                  {generatedPrompt ? (
                    <TextField
                      value={generatedPrompt}
                      fullWidth
                      multiline
                      minRows={2}
                      label="提示词（只读）"
                      InputProps={{ readOnly: true }}
                    />
                  ) : null}
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={handleGenerateAllSlots}
                      disabled={isWorking || !layoutConfig}
                    >
                      一键生成全部素材
                    </Button>
                    <Typography variant="caption" color="text.secondary">
                      已有素材将自动跳过
                    </Typography>
                  </Stack>
                </CardContent>
              </Card>
            </div>
          ) : null}

          {layoutConfig
            ? (layoutConfig.sections ?? []).map((section) => (
                <div key={`section-${section.id}`}>
                  <SectionCard
                    section={section as LayoutSection}
                    state={uiState}
                    onStateChange={handleStateChange}
                    onSlotUpload={handleSlotUpload}
                    onSlotGenerate={handleSlotGenerate}
                    onSlotRemoveBackground={handleSlotRemoveBackground}
                    disabled={isWorking}
                  />
                </div>
              ))
            : null}

          {layoutConfig ? (
            <div key="preview">
              <Card sx={{ height: '100%', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)' }}>
                <CardContent
                  sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%', overflowY: 'auto' }}
                >
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography variant="h6">预览卡片</Typography>
                      <Typography variant="caption" color="text.secondary">
                        素材与合成结果概览
                      </Typography>
                    </Box>
                    <Chip label={previewItems.length ? `已生成 ${previewItems.length} 层` : '待生成'} size="small" />
                  </Stack>

                  <Box
                    sx={{
                      position: 'relative',
                      width: '100%',
                      borderRadius: 2,
                      overflow: 'hidden',
                      border: '1px solid',
                      borderColor: 'grey.200',
                      pb: `${100 / parseAspectRatio(aspectRatio)}%`,
                      backgroundColor: '#f4f6f8',
                    }}
                  >
                    <Box
                      sx={{
                        position: 'absolute',
                        inset: 0,
                        backgroundImage: previewData.bgSlot?.uri
                          ? `url(${previewData.bgSlot.uri})`
                          : `url(${buildPlaceholderDataUrl('background', 720, 480)})`,
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                      }}
                    />
                    {previewData.subjects.slice(0, 4).map((slot, idx) => {
                      const positions = [
                        { left: '18%', top: '32%' },
                        { left: '68%', top: '30%' },
                        { left: '28%', top: '68%' },
                        { left: '68%', top: '66%' },
                      ];
                      const position = positions[idx] || positions[0];
                      return (
                        <Box
                          key={slot.id}
                          sx={{
                            position: 'absolute',
                            width: '30%',
                            aspectRatio: '1 / 1',
                            left: position.left,
                            top: position.top,
                            transform: 'translate(-50%, -50%)',
                            borderRadius: 1,
                            overflow: 'hidden',
                            border: '1px solid rgba(0,0,0,0.08)',
                            backgroundColor: '#fff',
                          }}
                        >
                          {slot.uri ? (
                            <Box
                              component="img"
                              src={slot.uri}
                              sx={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            />
                          ) : (
                            <Box
                              sx={{
                                width: '100%',
                                height: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: 12,
                                color: 'text.secondary',
                              }}
                            >
                              未生成
                            </Box>
                          )}
                        </Box>
                      );
                    })}
                  </Box>

                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {Object.entries(selectedOptions).flatMap(([k, values]) =>
                      values.map((v) => <Chip key={`${k}:${v}`} label={`${k}: ${v}`} size="small" />),
                    )}
                  </Stack>
                </CardContent>
              </Card>
            </div>
          ) : null}

          {showQuick ? (
            <div key="quick">
              <Card sx={{ height: '100%', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)' }}>
                <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography variant="h6">快速模式</Typography>
                      <Typography variant="caption" color="text.secondary">
                        单步生成，直接入画布
                      </Typography>
                    </Box>
                    <Chip label="单图" size="small" />
                  </Stack>
                  <TextField
                    label="快速提示词"
                    value={quickPrompt}
                    onChange={(e) => setQuickPrompt(e.target.value)}
                    fullWidth
                    multiline
                    minRows={2}
                  />
                  <Stack direction="row" spacing={1}>
                    <Button variant="contained" onClick={handleQuickGenerate} disabled={quickLoading}>
                      生成
                    </Button>
                    <Button variant="outlined" onClick={handleQuickAddToCanvas} disabled={!quickImage}>
                      送入画布
                    </Button>
                  </Stack>
                  {quickLoading ? <LinearProgress /> : null}
                  {quickStatus.message ? (
                    <Alert severity={quickStatus.type === 'error' ? 'error' : 'info'}>
                      {quickStatus.message}
                    </Alert>
                  ) : null}
                  {quickImage ? (
                    <Box
                      component="img"
                      src={quickImage}
                      alt="quick-result"
                      sx={{
                        width: '100%',
                        maxHeight: 260,
                        objectFit: 'cover',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'grey.200',
                      }}
                    />
                  ) : null}
                </CardContent>
              </Card>
            </div>
          ) : null}
        </ResponsiveGridLayout>
      </Box>

      <Box
        sx={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 12,
          px: 2,
        }}
      >
        <Paper
          elevation={4}
          sx={{
            p: 1.5,
            display: 'flex',
            gap: 2,
            alignItems: 'center',
            justifyContent: 'space-between',
            borderRadius: 2,
          }}
        >
          <Box sx={{ flex: 1 }}>
            {status.message ? (
              <Alert
                severity={status.type === 'error' ? 'error' : status.type === 'success' ? 'success' : 'info'}
                sx={{ py: 0.5 }}
              >
                {status.message}
              </Alert>
            ) : (
              <Typography variant="body2" color="text.secondary">
                选择完成后即可进入编辑流程，随时可回退调整。
              </Typography>
            )}
          </Box>
          <Box>{actionBar}</Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default SceneFlowPage;

const LoadingDots: React.FC = () => {
  return (
    <Box
      component="span"
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.6,
        ml: 1,
        verticalAlign: 'middle',
        '@keyframes chatDotsPulse': {
          '0%, 80%, 100%': { opacity: 0.3, transform: 'translateY(0)' },
          '40%': { opacity: 1, transform: 'translateY(-2px)' },
        },
        '& span': {
          width: 6,
          height: 6,
          borderRadius: '50%',
          backgroundColor: 'currentColor',
          display: 'inline-block',
          animation: 'chatDotsPulse 1.2s infinite ease-in-out',
        },
        '& span:nth-of-type(2)': {
          animationDelay: '0.2s',
        },
        '& span:nth-of-type(3)': {
          animationDelay: '0.4s',
        },
      }}
    >
      <Box component="span" />
      <Box component="span" />
      <Box component="span" />
    </Box>
  );
};

function normalizeForm(input: unknown): FormSection[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const sections: FormSection[] = [];
  input.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object') {
      return;
    }
    const maybeTitle = (raw as any).title;
    const title = typeof maybeTitle === 'string' && maybeTitle.trim() ? maybeTitle.trim() : `问题 ${index + 1}`;
    const maybeOptions = (raw as any).options;
    let options: string[] = [];
    if (Array.isArray(maybeOptions)) {
      options = maybeOptions.map((v) => String(v)).map((v) => v.trim()).filter(Boolean);
    } else if (typeof maybeOptions === 'string') {
      options = maybeOptions
        .split(/[,，、;\n]+/g)
        .map((v) => v.trim())
        .filter(Boolean);
    }
    options = Array.from(new Set(options)).slice(0, 6);
    if (options.length === 0) {
      return;
    }
    sections.push({ title, options });
  });
  return sections;
}

type LayoutItemSpec = { i: string; w: number; h: number };

function packGridItems(items: LayoutItemSpec[], cols: number): Layout[] {
  const colHeights = Array.from({ length: cols }, () => 0);
  return items.map((item) => {
    const width = Math.max(1, Math.min(cols, item.w));
    let bestX = 0;
    let bestY = Number.POSITIVE_INFINITY;
    for (let x = 0; x <= cols - width; x += 1) {
      let y = 0;
      for (let c = x; c < x + width; c += 1) {
        y = Math.max(y, colHeights[c]);
      }
      if (y < bestY) {
        bestY = y;
        bestX = x;
      }
    }
    for (let c = bestX; c < bestX + width; c += 1) {
      colHeights[c] = bestY + item.h;
    }
    return { i: item.i, x: bestX, y: bestY, w: width, h: item.h, static: true };
  });
}

function estimateSectionHeight(section: LayoutSection): number {
  const components = section.components ?? [];
  const types = components.map((comp) => comp.type);
  if (types.includes('media-uploader')) return 10;
  if (types.includes('prompt-editor')) return 8;
  if (components.length >= 5) return 7;
  if (components.length >= 3) return 6;
  return 5;
}

function coerceNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function initializeUiState(layoutConfig?: LayoutConfig): Record<string, any> {
  if (!layoutConfig?.sections?.length) return {};
  const next: Record<string, any> = {};
  layoutConfig.sections.forEach((section) => {
    (section.components ?? []).forEach((component) => {
      if (component.type === 'media-uploader') {
        const slots = Array.isArray(component.slots) ? component.slots : [];
        next[component.id] = slots.map((slot) => ({ ...slot }));
      } else if (component.type === 'multi-select') {
        next[component.id] = Array.isArray(component.default) ? component.default : [];
      } else if (component.type === 'toggle') {
        next[component.id] = Boolean(component.default);
      } else if (component.type === 'prompt-editor') {
        const fields = Array.isArray(component.fields) ? component.fields : [];
        const fieldState: Record<string, string> = {};
        fields.forEach((field: any) => {
          const fieldId = field?.id;
          if (fieldId) {
            fieldState[fieldId] = field?.default ?? '';
          }
        });
        next[component.id] = fieldState;
      } else if (component.type === 'number-input' || component.type === 'slider') {
        const rawDefault = component.default;
        const fallback = component.min ?? 0;
        const defaultValue = Number.isFinite(Number(rawDefault)) ? Number(rawDefault) : fallback;
        next[component.id] = defaultValue;
      } else if (component.type === 'color-palette' && component.allowMultiple) {
        next[component.id] = Array.isArray(component.default) ? component.default : [];
      } else {
        next[component.id] = component.default ?? '';
      }
    });
  });
  return next;
}

function extractMediaSlots(layoutConfig: LayoutConfig | undefined, uiState: Record<string, any>): MediaSlot[] {
  if (!layoutConfig?.sections?.length) return [];
  const slots: MediaSlot[] = [];
  layoutConfig.sections.forEach((section) => {
    (section.components ?? []).forEach((component) => {
      if (component.type === 'media-uploader') {
        const stateSlots = uiState[component.id];
        if (Array.isArray(stateSlots)) {
          stateSlots.forEach((slot) => slots.push(slot));
        }
      }
    });
  });
  return slots;
}

function parseAspectRatio(ratio: string): number {
  if (!ratio) return 3 / 4;
  if (ratio.includes(':')) {
    const [w, h] = ratio.split(':').map((v) => Number(v.trim()));
    if (w > 0 && h > 0) return w / h;
  }
  const num = Number(ratio);
  return Number.isFinite(num) && num > 0 ? num : 3 / 4;
}

function resolveSizeForSlot(isBackground: boolean, aspectRatio: string) {
  if (!isBackground) return { width: 512, height: 512 };
  const ratio = parseAspectRatio(aspectRatio);
  const width = 768;
  const height = Math.round(width / ratio);
  return { width, height };
}

function extractBase64(dataUrl: string): string {
  if (!dataUrl) return '';
  if (dataUrl.startsWith('data:') && dataUrl.includes(',')) {
    return dataUrl.split(',')[1] || '';
  }
  return dataUrl;
}

function mapToCanvas(input: {
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  bgW: number;
  bgH: number;
}) {
  const scale = TARGET_BG_WIDTH / input.bgW;
  const bgNewW = input.bgW * scale;
  const bgNewH = input.bgH * scale;
  const originX = CANVAS_DIMENSION / 2 - bgNewW / 2;
  const originY = CANVAS_DIMENSION / 2 - bgNewH / 2;
  return {
    x: originX + input.x * scale,
    y: originY + input.y * scale,
    width: input.width * scale,
    height: input.height * scale,
    zIndex: input.zIndex,
  };
}

function toApiErrorMessage(error: unknown, fallback: string) {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.error || error.response?.data?.message || error.message || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') resolve(reader.result);
      else reject(new Error('读取文件失败'));
    };
    reader.onerror = () => reject(new Error('读取文件失败'));
    reader.readAsDataURL(file);
  });
}

function buildPlaceholderDataUrl(label: string, width: number, height: number): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#f0f3f7"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#3c3c46" font-size="18" font-family="Arial">${escapeXml(
    label,
  )}</text></svg>`;
  const base64 = btoa(unescape(encodeURIComponent(svg)));
  return `data:image/svg+xml;base64,${base64}`;
}

function escapeXml(value: string) {
  return value.replace(/[<>&'"]/g, (c) => {
    switch (c) {
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '&':
        return '&amp;';
      case '"':
        return '&quot;';
      case "'":
        return '&apos;';
      default:
        return c;
    }
  });
}
