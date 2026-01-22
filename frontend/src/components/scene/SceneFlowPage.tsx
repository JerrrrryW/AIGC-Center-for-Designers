import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Divider,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Step,
  StepButton,
  Stepper,
  TextField,
  Typography,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api';
import DynamicRenderer, { type MediaSlot } from './dynamic/DynamicRenderer';

type ChatMessage = { role: 'user' | 'bot'; text: string; form?: FormSection[]; typing?: boolean };
type FormSection = { title: string; options: string[] };

type LayoutPayload = {
  layout_config?: any;
  layoutConfig?: any;
  text_response?: string;
  textResponse?: string;
};

type LayoutComponent = {
  id: string;
  type: string;
  default?: any;
  slots?: MediaSlot[];
  options?: any;
  fields?: any;
  min?: number;
  max?: number;
  step?: number;
  allowMultiple?: boolean;
};

type LayoutSection = {
  id: string;
  components?: LayoutComponent[];
};

type LayoutConfig = {
  meta?: Record<string, any>;
  sections?: LayoutSection[];
};

type StatusState = {
  type: 'idle' | 'loading' | 'success' | 'error';
  message: string;
};

const CANVAS_DIMENSION = 4000;
const TARGET_BG_WIDTH = 2400;
const FLOW_STEPS = ['对话澄清', '编辑与素材', '预览入画布'];
type ActiveCard = 'chat' | 'edit' | 'preview' | 'quick';
const CARD_ORDER: ActiveCard[] = ['chat', 'edit', 'preview', 'quick'];
const CARD_SPACING = 'clamp(220px, 28vw, 460px)';
const CARD_SIZES: Record<ActiveCard, { width: string; height: string }> = {
  chat: { width: 'min(92vw, 720px)', height: 'min(68vh, 600px)' },
  edit: { width: 'min(94vw, 820px)', height: 'min(72vh, 660px)' },
  preview: { width: 'min(90vw, 640px)', height: 'min(64vh, 560px)' },
  quick: { width: 'min(86vw, 520px)', height: 'min(58vh, 480px)' },
};

const SceneFlowPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeCard, setActiveCard] = useState<ActiveCard>('chat');
  const [status, setStatus] = useState<StatusState>({ type: 'idle', message: '' });

  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'bot', text: '你好！告诉我你想生成什么场景，我会先用选择题帮你把需求变清晰。' },
  ]);
  const [input, setInput] = useState('');
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string[]>>({});
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig | undefined>(undefined);
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [chatHistorySnapshot, setChatHistorySnapshot] = useState('');
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
  const activeCardIndex = useMemo(() => CARD_ORDER.indexOf(activeCard), [activeCard]);
  const activeStep = activeCard === 'edit' ? 1 : activeCard === 'preview' ? 2 : 0;

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

  const goToChat = useCallback(() => {
    setActiveCard('chat');
    setStatus({ type: 'idle', message: '' });
  }, []);

  const goToEdit = useCallback(() => {
    if (!layoutConfig) {
      setStatus({ type: 'error', message: '请先生成编辑界面。' });
      setActiveCard('chat');
      return;
    }
    setActiveCard('edit');
  }, [layoutConfig]);

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
      setChatHistorySnapshot(chatHistory);
      setUiState(initializeUiState(nextLayout));
      setStatus({ type: 'success', message: '编辑界面已就绪，开始准备素材吧。' });
      setActiveCard('edit');
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
      setActiveCard('preview');
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

  const actionDisabled = isSending || isWorking;
  const actionBar = useMemo(() => {
    if (activeCard === 'edit') {
      return (
        <Stack direction="row" spacing={1} alignItems="center">
          <Button variant="outlined" onClick={goToChat} disabled={actionDisabled}>
            返回聊天
          </Button>
          <Button
            variant="contained"
            onClick={handleGeneratePreviewToCanvas}
            disabled={actionDisabled || !layoutConfig}
          >
            生成预览
          </Button>
          <Button variant="text" onClick={() => setActiveCard('quick')} disabled={actionDisabled}>
            快速模式
          </Button>
        </Stack>
      );
    }
    if (activeCard === 'preview') {
      return (
        <Stack direction="row" spacing={1} alignItems="center">
          <Button variant="outlined" onClick={goToEdit} disabled={actionDisabled}>
            返回编辑
          </Button>
          <Button
            variant="contained"
            onClick={handleGoCanvas}
            disabled={actionDisabled || !previewItems.length}
          >
            进入画布
          </Button>
        </Stack>
      );
    }
    if (activeCard === 'quick') {
      return (
        <Stack direction="row" spacing={1} alignItems="center">
          <Button variant="outlined" onClick={() => setActiveCard('chat')}>
            回到流程
          </Button>
          <Button variant="contained" onClick={handleQuickGenerate} disabled={quickLoading}>
            快速生成
          </Button>
        </Stack>
      );
    }
    return (
      <Stack direction="row" spacing={1} alignItems="center">
        <Button
          variant="contained"
          onClick={handleGenerateEdit}
          disabled={isSending || !isFormComplete}
        >
          生成编辑界面
        </Button>
        <Button variant="text" onClick={() => setActiveCard('quick')} disabled={isSending}>
          快速模式
        </Button>
      </Stack>
    );
  }, [
    actionDisabled,
    activeCard,
    goToChat,
    goToEdit,
    handleGenerateEdit,
    handleGeneratePreviewToCanvas,
    handleGoCanvas,
    handleQuickGenerate,
    isFormComplete,
    isSending,
    layoutConfig,
    previewItems.length,
    quickLoading,
  ]);

  const cardStyleFor = useCallback(
    (index: number) => {
      const offset = index - activeCardIndex;
      const translateX = `calc(${offset} * ${CARD_SPACING})`;
      const isActive = offset === 0;
      const isHidden = Math.abs(offset) >= 2;
      return {
        position: 'absolute' as const,
        top: '50%',
        left: '50%',
        transform: `translate(-50%, -50%) translateX(${translateX}) scale(${isActive ? 1 : 0.92})`,
        opacity: isHidden ? 0 : isActive ? 1 : 0.35,
        pointerEvents: isActive ? 'auto' : 'none',
        transition: 'transform 420ms ease, opacity 320ms ease',
        zIndex: 20 - Math.abs(offset),
      };
    },
    [activeCardIndex],
  );

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
        {FLOW_STEPS.map((label, index) => (
          <Step key={label}>
            {index === 0 ? (
              <StepButton onClick={goToChat}>{label}</StepButton>
            ) : index === 1 ? (
              <StepButton onClick={goToEdit}>{label}</StepButton>
            ) : (
              <StepButton
                onClick={() => {
                  if (!previewItems.length) {
                    setStatus({ type: 'error', message: '请先生成预览。' });
                    setActiveCard('edit');
                    return;
                  }
                  setActiveCard('preview');
                }}
              >
                {label}
              </StepButton>
            )}
          </Step>
        ))}
      </Stepper>

      <Box sx={{ position: 'relative', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        {CARD_ORDER.map((key, index) => {
          const size = CARD_SIZES[key];
          return (
            <Box key={key} sx={cardStyleFor(index)}>
              <Card
                sx={{
                  width: size.width,
                  height: size.height,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  boxShadow: '0 24px 80px rgba(15, 23, 42, 0.14)',
                }}
              >
                {key === 'chat' ? (
                  <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography variant="h6">步骤 1：对话澄清</Typography>
                        <Typography variant="caption" color="text.secondary">
                          多轮对话 + 选择题，明确需求
                        </Typography>
                      </Box>
                      <Chip label={isFormComplete ? '可进入编辑' : '待完成'} size="small" />
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
                              {!msg.typing && msg.form?.length ? (
                                <Box sx={{ mt: 1 }}>
                                  <Card variant="outlined" sx={{ bgcolor: 'background.paper' }}>
                                    <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                      <Typography variant="subtitle2" gutterBottom>
                                        选择题表单
                                      </Typography>
                                      <Stack spacing={1.5}>
                                        {normalizeForm(msg.form).map((section, sectionIndex) => (
                                          <Box key={`${section.title}-${sectionIndex}`}>
                                            <Typography
                                              variant="caption"
                                              sx={{ opacity: 0.9, display: 'block' }}
                                            >
                                              {section.title}
                                            </Typography>
                                            <Stack
                                              direction="row"
                                              spacing={1}
                                              flexWrap="wrap"
                                              useFlexGap
                                              sx={{ mt: 0.5 }}
                                            >
                                              {(Array.isArray(section.options)
                                                ? section.options
                                                : []
                                              ).map((opt) => {
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
                                        {msg.form.length > 0 && !isFormComplete ? (
                                          <Typography variant="caption" color="error">
                                            请为每项至少选择一个选项，才能进入编辑界面。
                                          </Typography>
                                        ) : null}
                                      </Stack>
                                    </CardContent>
                                  </Card>
                                </Box>
                              ) : null}
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
                ) : null}

                {key === 'edit' ? (
                  <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography variant="h6">步骤 2：编辑与素材</Typography>
                        <Typography variant="caption" color="text.secondary">
                          配置素材与表单，准备预览
                        </Typography>
                      </Box>
                      <Chip label={layoutConfig ? '已就绪' : '待生成'} size="small" />
                    </Stack>
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
                    <Typography variant="body2" color="text.secondary">
                      画幅：{aspectRatio}
                    </Typography>
                    <Divider />

                    <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto', pr: 1 }}>
                      {layoutConfig ? (
                        <>
                          <Typography variant="subtitle2" gutterBottom>
                            生成目标概要
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {layoutConfig.meta?.summary || '在下方配置素材，完成后生成预览。'}
                          </Typography>
                          {generatedPrompt ? (
                            <TextField
                              value={generatedPrompt}
                              fullWidth
                              multiline
                              minRows={3}
                              label="提示词（只读）"
                              InputProps={{ readOnly: true }}
                            />
                          ) : null}
                          <Divider sx={{ my: 2 }} />
                          <DynamicRenderer
                            config={layoutConfig}
                            state={uiState}
                            onStateChange={handleStateChange}
                            onSlotUpload={handleSlotUpload}
                            onSlotGenerate={handleSlotGenerate}
                            onSlotRemoveBackground={handleSlotRemoveBackground}
                            disabled={isWorking}
                          />
                        </>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          请先完成对话并生成编辑界面。
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                ) : null}

                {key === 'preview' ? (
                  <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography variant="h6">步骤 3：预览与入画布</Typography>
                        <Typography variant="caption" color="text.secondary">
                          查看合成效果并导入画布
                        </Typography>
                      </Box>
                      <Chip
                        label={previewItems.length ? `已生成 ${previewItems.length} 层` : '待生成'}
                        size="small"
                      />
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

                    <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto', pr: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        当前选择
                      </Typography>
                      {Object.keys(selectedOptions).length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          暂无
                        </Typography>
                      ) : (
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                          {Object.entries(selectedOptions).flatMap(([k, values]) =>
                            values.map((v) => <Chip key={`${k}:${v}`} label={`${k}: ${v}`} size="small" />),
                          )}
                        </Stack>
                      )}

                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>
                        素材准备情况
                      </Typography>
                      {mediaSlots.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          还没有素材槽位。
                        </Typography>
                      ) : (
                        <Stack spacing={1}>
                          {mediaSlots.map((slot) => (
                            <Box key={slot.id}>
                              <Typography variant="subtitle2">
                                {slot.label} {slot.uri ? '✅' : '—'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {slot.uri ? (slot.hasTransparentBg ? '已抠图' : '未抠图') : '未准备'}
                              </Typography>
                            </Box>
                          ))}
                        </Stack>
                      )}

                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>
                        对话记录
                      </Typography>
                      <TextField
                        value={chatHistorySnapshot || chatHistory}
                        fullWidth
                        multiline
                        minRows={4}
                        InputProps={{ readOnly: true }}
                      />
                    </Box>
                  </CardContent>
                ) : null}

                {key === 'quick' ? (
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
                ) : null}
              </Card>
            </Box>
          );
        })}
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
