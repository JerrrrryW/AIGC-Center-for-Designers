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

type ChatMessage = { role: 'user' | 'bot'; text: string; questions?: QuestionItem[]; typing?: boolean };

type QuestionType = 'single' | 'multi' | 'text';

type QuestionItem = {
  id: string;
  title: string;
  type: QuestionType;
  options?: string[];
  priority?: number;
  allowSkip?: boolean;
  placeholder?: string;
};

type AnswerValue = string | string[];

type AnswerState = {
  value?: AnswerValue;
  skipped?: boolean;
  updatedAt?: number;
};

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
  const [answers, setAnswers] = useState<Record<string, AnswerState>>({});
  const [lastCommittedAnswers, setLastCommittedAnswers] = useState<Record<string, AnswerState>>({});
  const [boardDirtyMap, setBoardDirtyMap] = useState<Record<string, boolean>>({});
  const [editingMap, setEditingMap] = useState<Record<string, boolean>>({});
  const [textDrafts, setTextDrafts] = useState<Record<string, string>>({});
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const sendingRef = useRef(false);
  const pendingChatQueueRef = useRef<Array<{ text: string; forceForm?: boolean }>>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  const answersRef = useRef<Record<string, AnswerState>>({});
  const questionListRef = useRef<QuestionItem[]>([]);

  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig | undefined>(undefined);
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [isWorking, setIsWorking] = useState(false);
  const [uiState, setUiState] = useState<Record<string, any>>({});
  const [previewItems, setPreviewItems] = useState<any[]>([]);

  const [quickPrompt, setQuickPrompt] = useState('');
  const [quickImage, setQuickImage] = useState<string | null>(null);
  const [quickStatus, setQuickStatus] = useState<StatusState>({ type: 'idle', message: '' });
  const [quickLoading, setQuickLoading] = useState(false);

  const questionContext = useMemo(() => {
    const map = new Map<string, QuestionItem>();
    const order: string[] = [];
    messages.forEach((msg) => {
      (msg.questions ?? []).forEach((question) => {
        if (!map.has(question.id)) {
          order.push(question.id);
        }
        map.set(question.id, question);
      });
    });
    return { map, order };
  }, [messages]);

  const questionList = useMemo(
    () => questionContext.order.map((id) => questionContext.map.get(id)).filter(Boolean) as QuestionItem[],
    [questionContext],
  );

  const selectedOptions = useMemo(
    () => buildSelectedOptionsPayload(questionList, answers),
    [answers, questionList],
  );

  const pendingQuestions = useMemo(() => {
    const orderIndex = new Map(questionContext.order.map((id, idx) => [id, idx]));
    return questionList
      .filter((question) => !isAnswerResolved(answers[question.id]))
      .sort((a, b) => {
        const priorityA = a.priority ?? 3;
        const priorityB = b.priority ?? 3;
        if (priorityA !== priorityB) return priorityA - priorityB;
        return (orderIndex.get(a.id) ?? 0) - (orderIndex.get(b.id) ?? 0);
      });
  }, [answers, questionContext.order, questionList]);

  const confirmedQuestions = useMemo(
    () => questionList.filter((question) => isAnswerResolved(answers[question.id])),
    [answers, questionList],
  );

  const hasQuestions = questionList.length > 0;

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

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    answersRef.current = answers;
  }, [answers]);

  useEffect(() => {
    questionListRef.current = questionList;
  }, [questionList]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  const sendChatMessage = useCallback(
    async (text: string, options?: { forceForm?: boolean }) => {
      const value = text.trim();
      if (!value) return;
      if (sendingRef.current) {
        pendingChatQueueRef.current.push({ text: value, forceForm: options?.forceForm });
        return;
      }
      sendingRef.current = true;
      setIsSending(true);
      setStatus({ type: 'loading', message: '正在生成对话回复...' });
      const nextMessages = [...messagesRef.current, { role: 'user' as const, text: value }];
      setMessages([...nextMessages, { role: 'bot', text: 'Agent 输入中...', typing: true }]);
      queueMicrotask(scrollToBottom);

      const answersSnapshot = cloneAnswerState(answersRef.current);
      const selectedSnapshot = buildSelectedOptionsPayload(questionListRef.current, answersSnapshot);
      try {
        const res = await api.post('/chat', {
          messages: nextMessages.map((m) => ({
            role: m.role === 'bot' ? 'assistant' : 'user',
            content: m.text,
          })),
          selected_options: selectedSnapshot,
          force_form: Boolean(options?.forceForm),
        });
        const obj = res.data?.json_object ?? res.data;
        const botText = obj?.text_response ?? '好的，我们继续。';
        const questions = normalizeQuestions(obj?.form);
        setMessages((prev) => {
          const withoutTyping = prev.filter((m) => !m.typing);
          return [...withoutTyping, { role: 'bot', text: botText, questions }];
        });
        setStatus({ type: 'success', message: '对话已更新，可继续补充或进入编辑。' });
        setLastCommittedAnswers(answersSnapshot);
        setBoardDirtyMap({});
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
        sendingRef.current = false;
        setIsSending(false);
        const next = pendingChatQueueRef.current.shift();
        if (next) {
          setTimeout(() => {
            void sendChatMessage(next.text, { forceForm: next.forceForm });
          }, 0);
        }
        queueMicrotask(scrollToBottom);
      }
    },
    [scrollToBottom],
  );

  const setAnswerForQuestion = useCallback(
    (
      question: QuestionItem,
      nextValue: AnswerValue | undefined,
      source: 'chat' | 'board',
      skipped = false,
      triggerFollowup = true,
    ) => {
      const normalizedValue = normalizeAnswerValue(nextValue);
      const nextState: AnswerState = {
        value: normalizedValue,
        skipped,
        updatedAt: Date.now(),
      };
      setAnswers((prev) => ({ ...prev, [question.id]: nextState }));
      answersRef.current = { ...answersRef.current, [question.id]: nextState };

      if (source === 'board') {
        setBoardDirtyMap((prev) => {
          const next = { ...prev };
          const committed = lastCommittedAnswers[question.id];
          if (isAnswerEqual(nextState, committed)) {
            delete next[question.id];
          } else {
            next[question.id] = true;
          }
          return next;
        });
      }

      if (source === 'chat' && triggerFollowup) {
        const label = formatAnswerLabel(nextState);
        if (label) {
          void sendChatMessage(`已确认：${question.title} = ${label}`, { forceForm: true });
        }
      }
    },
    [lastCommittedAnswers, sendChatMessage],
  );

  const handleOptionSelect = useCallback(
    (question: QuestionItem, option: string, source: 'chat' | 'board') => {
      const current = answers[question.id]?.value;
      if (question.type === 'multi') {
        const currentList = toAnswerList(current);
        const nextList = currentList.includes(option)
          ? currentList.filter((v) => v !== option)
          : [...currentList, option];
        const shouldTrigger = source !== 'chat';
        setAnswerForQuestion(
          question,
          nextList.length ? nextList : undefined,
          source,
          false,
          shouldTrigger,
        );
        return;
      }
      setAnswerForQuestion(question, option, source, false);
    },
    [answers, setAnswerForQuestion],
  );

  const handleSkipQuestion = useCallback(
    (question: QuestionItem, source: 'chat' | 'board') => {
      setAnswerForQuestion(question, undefined, source, true);
    },
    [setAnswerForQuestion],
  );

  const handleTextSubmit = useCallback(
    (question: QuestionItem, source: 'chat' | 'board') => {
      const draft = (textDrafts[question.id] ?? '').trim();
      if (!draft) return;
      setAnswerForQuestion(question, draft, source, false);
      setTextDrafts((prev) => ({ ...prev, [question.id]: '' }));
    },
    [setAnswerForQuestion, textDrafts],
  );

  const handleSend = useCallback(async () => {
    const value = input.trim();
    if (!value || isSending) return;
    setInput('');
    await sendChatMessage(value);
  }, [input, isSending, sendChatMessage]);

  const boardDirtyIds = useMemo(() => Object.keys(boardDirtyMap), [boardDirtyMap]);
  const hasBoardChanges = boardDirtyIds.length > 0;

  const handleApplyBoardUpdates = useCallback(async () => {
    if (!hasBoardChanges || isSending) return;
    const changes = boardDirtyIds
      .map((id) => {
        const question = questionContext.map.get(id) ?? {
          id,
          title: id,
          type: 'text' as QuestionType,
        };
        const nextState = answers[id];
        const label = formatAnswerLabel(nextState);
        return { title: question.title, label };
      })
      .filter((item) => item.label);
    if (!changes.length) return;
    const summary = changes.map((item) => `将「${item.title}」修改为「${item.label}」`).join('；');
    await sendChatMessage(summary, { forceForm: true });
  }, [answers, boardDirtyIds, hasBoardChanges, isSending, questionContext.map, sendChatMessage]);

  const toggleEditing = useCallback((questionId: string) => {
    setEditingMap((prev) => ({ ...prev, [questionId]: !prev[questionId] }));
  }, []);

  const handleGenerateEdit = useCallback(async () => {
    if (isSending) return;
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
  }, [chatHistory, isSending, messages, selectedOptions]);

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

  const handleConfirmMultiQuestion = useCallback(
    (question: QuestionItem) => {
      const current = answersRef.current[question.id];
      const label = formatAnswerLabel(current);
      if (!label) return;
      setAnswerForQuestion(question, current?.value, 'chat', Boolean(current?.skipped), true);
    },
    [setAnswerForQuestion],
  );

  const pendingCount = pendingQuestions.length;
  const confirmedCount = confirmedQuestions.length;

  const renderQuestionControls = (question: QuestionItem, source: 'chat' | 'board') => {
    const answerState = answers[question.id];
    const selectedValues = toAnswerList(answerState?.value);
    const allowSkip = question.allowSkip !== false;
    const isSkipped = Boolean(answerState?.skipped);
    const isDisabled = isSending;

    if (question.type === 'text') {
      const draftValue =
        textDrafts[question.id] ??
        (typeof answerState?.value === 'string' ? answerState.value : '');
      return (
        <Stack spacing={1}>
          <TextField
            value={draftValue}
            onChange={(e) =>
              setTextDrafts((prev) => ({ ...prev, [question.id]: e.target.value }))
            }
            size="small"
            placeholder={question.placeholder || '请输入补充描述'}
            disabled={isDisabled}
            multiline
            minRows={2}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                handleTextSubmit(question, source);
              }
            }}
          />
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              variant="contained"
              onClick={() => handleTextSubmit(question, source)}
              disabled={isDisabled || !(textDrafts[question.id] ?? '').trim()}
            >
              确认
            </Button>
            {allowSkip ? (
              <Button
                size="small"
                variant="text"
                onClick={() => handleSkipQuestion(question, source)}
                disabled={isDisabled}
              >
                暂不确定
              </Button>
            ) : null}
          </Stack>
        </Stack>
      );
    }

    if (question.type === 'multi' && source === 'chat') {
      return (
        <Stack spacing={1}>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {(question.options ?? []).map((opt) => {
              const selected = selectedValues.includes(opt);
              return (
                <Chip
                  key={`${question.id}-${opt}`}
                  label={opt}
                  size="small"
                  clickable
                  color={selected ? 'primary' : 'default'}
                  variant={selected ? 'filled' : 'outlined'}
                  onClick={() => handleOptionSelect(question, opt, source)}
                  disabled={isDisabled}
                />
              );
            })}
            {allowSkip ? (
              <Chip
                label="暂不确定"
                size="small"
                color={isSkipped ? 'primary' : 'default'}
                variant={isSkipped ? 'filled' : 'outlined'}
                onClick={() => handleSkipQuestion(question, source)}
                disabled={isDisabled}
              />
            ) : null}
          </Stack>
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              variant="contained"
              onClick={() => handleConfirmMultiQuestion(question)}
              disabled={isDisabled || selectedValues.length === 0}
            >
              确认
            </Button>
          </Stack>
        </Stack>
      );
    }

    return (
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {(question.options ?? []).map((opt) => {
          const selected = selectedValues.includes(opt);
          return (
            <Chip
              key={`${question.id}-${opt}`}
              label={opt}
              size="small"
              clickable
              color={selected ? 'primary' : 'default'}
              variant={selected ? 'filled' : 'outlined'}
              onClick={() => handleOptionSelect(question, opt, source)}
              disabled={isDisabled}
            />
          );
        })}
        {allowSkip ? (
          <Chip
            label="暂不确定"
            size="small"
            color={isSkipped ? 'primary' : 'default'}
            variant={isSkipped ? 'filled' : 'outlined'}
            onClick={() => handleSkipQuestion(question, source)}
            disabled={isDisabled}
          />
        ) : null}
      </Stack>
    );
  };

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
          <Button variant="contained" onClick={handleGenerateEdit} disabled={isSending}>
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
                    label={
                      hasQuestions
                        ? pendingCount > 0
                          ? `待澄清 ${pendingCount}`
                          : '可进入编辑'
                        : '等待描述'
                    }
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
                          {msg.role === 'bot' && msg.questions?.length ? (
                            <Stack spacing={1} sx={{ mt: 1 }}>
                              {msg.questions.map((question) => (
                                <Box
                                  key={question.id}
                                  sx={{
                                    p: 1,
                                    borderRadius: 1.5,
                                    border: '1px solid rgba(15, 23, 42, 0.12)',
                                    backgroundColor: '#fff',
                                  }}
                                >
                                  <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                                    {question.title}
                                  </Typography>
                                  {renderQuestionControls(question, 'chat')}
                                </Box>
                              ))}
                            </Stack>
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
            </Card>
          </div>

          <div key="form">
            <Card sx={{ height: '100%', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)' }}>
              <CardContent
                sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%', overflowY: 'auto' }}
              >
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6">意图板</Typography>
                    <Typography variant="caption" color="text.secondary">
                      对话中的意图将在此汇总
                    </Typography>
                  </Box>
                  <Stack direction="row" spacing={1} alignItems="center">
                    {hasBoardChanges ? (
                      <Button size="small" variant="contained" onClick={handleApplyBoardUpdates} disabled={isSending}>
                        更新{boardDirtyIds.length ? `（${boardDirtyIds.length}）` : ''}
                      </Button>
                    ) : null}
                    <Chip
                      label={pendingCount > 0 ? `待确认 ${pendingCount}` : '已同步'}
                      size="small"
                    />
                  </Stack>
                </Stack>

                <Stack spacing={2}>
                  <Box>
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                      <Typography variant="subtitle2">待确认</Typography>
                      <Chip label={pendingCount} size="small" />
                    </Stack>
                    {pendingQuestions.length ? (
                      <Stack spacing={1.5} sx={{ mt: 1 }}>
                        {pendingQuestions.map((question) => (
                          <Box key={question.id}>
                            <Typography variant="caption" sx={{ opacity: 0.9, display: 'block' }}>
                              {question.title}
                            </Typography>
                            {renderQuestionControls(question, 'board')}
                          </Box>
                        ))}
                      </Stack>
                    ) : (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        暂无待确认问题。
                      </Typography>
                    )}
                  </Box>

                  <Box>
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                      <Typography variant="subtitle2">已确认</Typography>
                      <Chip label={confirmedCount} size="small" />
                    </Stack>
                    {confirmedQuestions.length ? (
                      <Stack spacing={1.2} sx={{ mt: 1 }}>
                        {confirmedQuestions.map((question) => {
                          const answerState = answers[question.id];
                          const answerLabel = formatAnswerLabel(answerState);
                          const isEditing = Boolean(editingMap[question.id]);
                          return (
                            <Box
                              key={question.id}
                              sx={{
                                p: 1,
                                borderRadius: 1.5,
                                border: '1px solid rgba(15, 23, 42, 0.1)',
                                backgroundColor: 'rgba(15, 23, 42, 0.02)',
                              }}
                            >
                              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                                <Typography variant="body2">{question.title}</Typography>
                                <Button size="small" variant="text" onClick={() => toggleEditing(question.id)}>
                                  {isEditing ? '收起' : '改'}
                                </Button>
                              </Stack>
                              <Typography variant="caption" color="text.secondary">
                                {answerLabel || '暂未填写'}
                              </Typography>
                              {isEditing ? <Box sx={{ mt: 1 }}>{renderQuestionControls(question, 'board')}</Box> : null}
                            </Box>
                          );
                        })}
                      </Stack>
                    ) : (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        还没有已确认的选项。
                      </Typography>
                    )}
                  </Box>
                </Stack>
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
                可随时进入编辑流程；对话可继续无限追问细化需求。
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

function normalizeQuestions(input: unknown): QuestionItem[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const questions: QuestionItem[] = [];
  input.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object') {
      return;
    }
    const rawRecord = raw as Record<string, any>;
    const rawTitle = rawRecord.title;
    const title = typeof rawTitle === 'string' && rawTitle.trim() ? rawTitle.trim() : `问题 ${index + 1}`;
    const rawId = rawRecord.id ?? rawRecord.key ?? rawRecord.question_id;
    const questionId = normalizeQuestionId(rawId, title, index);
    const rawType = String(rawRecord.type ?? rawRecord.question_type ?? '').trim().toLowerCase();
    const multiple = rawRecord.multiple === true || rawType === 'multi' || rawType === 'multiple';
    const isText =
      rawType === 'text' ||
      rawType === 'text-input' ||
      rawType === 'input' ||
      rawType === 'free-text';
    const type: QuestionType = isText ? 'text' : multiple ? 'multi' : 'single';
    const allowSkip =
      typeof rawRecord.allow_skip === 'boolean'
        ? rawRecord.allow_skip
        : typeof rawRecord.allowSkip === 'boolean'
          ? rawRecord.allowSkip
          : true;
    const priorityRaw = Number(rawRecord.priority);
    const priority = Number.isFinite(priorityRaw)
      ? Math.min(5, Math.max(1, Math.round(priorityRaw)))
      : 3;
    const placeholder =
      typeof rawRecord.placeholder === 'string' && rawRecord.placeholder.trim()
        ? rawRecord.placeholder.trim()
        : typeof rawRecord.helperText === 'string'
          ? rawRecord.helperText.trim()
          : undefined;

    const options = parseOptions(rawRecord.options);
    if (type !== 'text' && options.length === 0) {
      return;
    }
    questions.push({
      id: questionId,
      title,
      type,
      options: type === 'text' ? undefined : options,
      priority,
      allowSkip,
      placeholder,
    });
  });
  return questions;
}

function parseOptions(value: unknown): string[] {
  let options: string[] = [];
  if (Array.isArray(value)) {
    options = value.map((v) => String(v)).map((v) => v.trim()).filter(Boolean);
  } else if (typeof value === 'string') {
    options = value
      .split(/[,，、;\n]+/g)
      .map((v) => v.trim())
      .filter(Boolean);
  }
  return Array.from(new Set(options)).slice(0, 8);
}

function normalizeQuestionId(rawId: unknown, title: string, index: number): string {
  const base = String(rawId ?? title ?? `question-${index + 1}`).trim().toLowerCase();
  const slug = base
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^[-_]+|[-_]+$/g, '');
  if (slug) return slug;
  const hash = Math.abs(hashString(base || title)).toString(36).slice(0, 6);
  return `question-${index + 1}-${hash || 'id'}`;
}

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

function normalizeAnswerValue(value?: AnswerValue): AnswerValue | undefined {
  if (Array.isArray(value)) {
    const next = value.map((v) => String(v).trim()).filter(Boolean);
    return next.length ? Array.from(new Set(next)) : undefined;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? trimmed : undefined;
  }
  return undefined;
}

function toAnswerList(value?: AnswerValue): string[] {
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') return value ? [value] : [];
  return [];
}

function isAnswerResolved(state?: AnswerState): boolean {
  if (!state) return false;
  if (state.skipped) return true;
  const value = normalizeAnswerValue(state.value);
  if (Array.isArray(value)) return value.length > 0;
  return typeof value === 'string' && value.length > 0;
}

function formatAnswerLabel(state?: AnswerState): string {
  if (!state) return '';
  if (state.skipped) return '暂不确定';
  const value = normalizeAnswerValue(state.value);
  if (Array.isArray(value)) {
    return value.join(' / ');
  }
  if (typeof value === 'string') return value;
  return '';
}

function isAnswerEqual(a?: AnswerState, b?: AnswerState): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (Boolean(a.skipped) !== Boolean(b.skipped)) return false;
  const listA = toAnswerList(normalizeAnswerValue(a.value));
  const listB = toAnswerList(normalizeAnswerValue(b.value));
  if (listA.length !== listB.length) return false;
  return listA.every((value, idx) => value === listB[idx]);
}

function cloneAnswerState(state: Record<string, AnswerState>): Record<string, AnswerState> {
  const next: Record<string, AnswerState> = {};
  Object.entries(state).forEach(([key, value]) => {
    const cloned: AnswerState = { ...value };
    if (Array.isArray(value.value)) {
      cloned.value = [...value.value];
    }
    next[key] = cloned;
  });
  return next;
}

function buildSelectedOptionsPayload(
  questions: QuestionItem[],
  state: Record<string, AnswerState>,
): Record<string, string[]> {
  const payload: Record<string, string[]> = {};
  questions.forEach((question) => {
    const answerState = state[question.id];
    if (!answerState || answerState.skipped) return;
    const values = toAnswerList(normalizeAnswerValue(answerState.value));
    if (!values.length) return;
    payload[question.title] = values;
  });
  return payload;
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
