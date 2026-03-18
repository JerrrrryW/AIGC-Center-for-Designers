import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Backdrop,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  CircularProgress,
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
import { type LayoutConfig, type LayoutSection, type MediaSlot } from './dynamic/DynamicRenderer';
import SceneEditWorkspace from './edit/SceneEditWorkspace';
import { type SceneDraft, type SceneMaterialDraftSlot } from './edit/sceneDraft';

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
  layer_plan?: any;
  layerPlan?: any;
  draft?: Partial<SceneDraft> & {
    controls?: Partial<SceneDraft['controls']> & {
      positivePrompt?: string;
    };
  };
  text_response?: string;
  textResponse?: string;
};

type StatusState = {
  type: 'idle' | 'loading' | 'success' | 'error';
  message: string;
};

type TextLayerDraft = {
  id: string;
  name: string;
  text: string;
  role: 'title' | 'subtitle' | 'body';
  color?: string;
  align?: 'left' | 'center' | 'right';
};

type PreviewComposition = {
  layers: any[];
  canvasWidth: number;
  canvasHeight: number;
  updatedAt: number;
};

const CANVAS_DIMENSION = 4000;
const TARGET_BG_WIDTH = 2400;
const FLOW_STEPS = ['对话澄清', '编辑与素材', '预览入画布'];
const GRID_BREAKPOINTS = { lg: 1200, md: 900, sm: 600, xs: 0 };
const GRID_COLS = { lg: 12, md: 8, sm: 6, xs: 4 };
const GRID_ROW_HEIGHT = 32;
const GRID_MARGIN: [number, number] = [16, 16];
const GRID_CONTAINER_PADDING: [number, number] = [8, 8];

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
  const gridContainerRef = useRef<HTMLDivElement | null>(null);
  const sendingRef = useRef(false);
  const pendingChatQueueRef = useRef<Array<{ text: string; forceForm?: boolean }>>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  const answersRef = useRef<Record<string, AnswerState>>({});
  const questionListRef = useRef<QuestionItem[]>([]);
  const previewRequestRef = useRef(0);

  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig | undefined>(undefined);
  const [layerPlan, setLayerPlan] = useState<any>(null);
  const [sceneDraft, setSceneDraft] = useState<SceneDraft | undefined>(undefined);
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [isWorking, setIsWorking] = useState(false);
  const [uiState, setUiState] = useState<Record<string, any>>({});
  const [previewItems, setPreviewItems] = useState<any[]>([]);
  const [previewComposition, setPreviewComposition] = useState<PreviewComposition | null>(null);
  const [isPreviewSyncing, setIsPreviewSyncing] = useState(false);
  const [previewError, setPreviewError] = useState('');

  const [quickPrompt, setQuickPrompt] = useState('');
  const [quickImage, setQuickImage] = useState<string | null>(null);
  const [quickStatus, setQuickStatus] = useState<StatusState>({ type: 'idle', message: '' });
  const [quickLoading, setQuickLoading] = useState(false);
  const [step1Rows, setStep1Rows] = useState(0);

  const effectiveLayoutConfig = useMemo(
    () => ensureStableEditSections(layoutConfig, layerPlan, sceneDraft),
    [layoutConfig, layerPlan, sceneDraft],
  );

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
    const ratioField = effectiveLayoutConfig?.meta?.aspect_ratio_field;
    const fromUi = ratioField ? uiState?.[ratioField] : undefined;
    const fromSelected = selectedOptions['画幅比例']?.[0];
    const fromMeta = effectiveLayoutConfig?.meta?.aspect_ratio;
    return fromUi || fromSelected || fromMeta || '3:4';
  }, [effectiveLayoutConfig?.meta?.aspect_ratio, effectiveLayoutConfig?.meta?.aspect_ratio_field, selectedOptions, uiState]);

  const mediaSlots = useMemo(
    () => extractMediaSlots(effectiveLayoutConfig, uiState),
    [effectiveLayoutConfig, uiState],
  );
  const textLayers = useMemo(
    () => collectTextLayers(effectiveLayoutConfig, uiState, selectedOptions),
    [effectiveLayoutConfig, selectedOptions, uiState],
  );
  const activeStep = previewComposition ? 2 : effectiveLayoutConfig ? 1 : 0;

  useEffect(() => {
    if (!effectiveLayoutConfig) return;
    setUiState((prev) => {
      const defaults = buildSceneUiState(effectiveLayoutConfig, sceneDraft);
      if (!Object.keys(defaults).length) {
        return prev;
      }
      const next = { ...defaults, ...prev };
      const prevKeys = Object.keys(prev);
      const nextKeys = Object.keys(next);
      if (prevKeys.length === nextKeys.length && prevKeys.every((key) => prev[key] === next[key])) {
        return prev;
      }
      return next;
    });
  }, [effectiveLayoutConfig, sceneDraft]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (effectiveLayoutConfig) return;
    const el = gridContainerRef.current;
    if (!el) return;
    const updateRows = () => {
      const height = el.clientHeight;
      if (!height) return;
      const adjustedHeight = Math.max(0, height - GRID_CONTAINER_PADDING[1] * 2);
      const rowUnit = GRID_ROW_HEIGHT + GRID_MARGIN[1];
      const rows = Math.max(1, Math.floor((adjustedHeight + GRID_MARGIN[1]) / rowUnit));
      setStep1Rows((prev) => (prev === rows ? prev : rows));
    };
    updateRows();
    const observer = new ResizeObserver(updateRows);
    observer.observe(el);
    return () => observer.disconnect();
  }, [effectiveLayoutConfig]);

  useEffect(() => {
    answersRef.current = answers;
  }, [answers]);

  useEffect(() => {
    questionListRef.current = questionList;
  }, [questionList]);

  const buildPendingItemsFromComposition = useCallback(
    (composition: PreviewComposition) => {
      const bgSlot = mediaSlots.find((s) => s.layerType === 'background');
      const bgUri = bgSlot?.uri;
      const bgW = composition.canvasWidth;
      const bgH = composition.canvasHeight;
      const paletteColor = findPrimaryColor(effectiveLayoutConfig, uiState) || '#111827';

      const pending: any[] = [];
      const backgroundDataUrl =
        bgUri ?? buildPlaceholderDataUrl(`background:${generatedPrompt || ''}`, bgW, bgH);
      pending.push({
        kind: 'image',
        dataUrl: backgroundDataUrl,
        name: '背景',
        ...mapToCanvas({
          x: 0,
          y: 0,
          width: bgW,
          height: bgH,
          zIndex: 0,
          bgW,
          bgH,
        }),
      });

      composition.layers.forEach((layer: any) => {
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
        if ((layer.kind || 'image') === 'image') {
          const dataUrl = layer.image_base64?.startsWith('data:')
            ? layer.image_base64
            : `data:image/png;base64,${layer.image_base64}`;
          pending.push({
            kind: 'image',
            dataUrl,
            name: layer.name || layer.id || '图层',
            ...mapped,
          });
          return;
        }
        pending.push({
          kind: layer.kind,
          name: layer.name || layer.id || '图层',
          text: layer.text,
          style: {
            color: layer.style?.color || paletteColor,
            fontSize: layer.style?.fontSize,
            fontWeight: layer.style?.fontWeight,
            align: layer.style?.align,
            backgroundColor: layer.style?.backgroundColor,
            padding: layer.style?.padding,
            lineHeight: layer.style?.lineHeight,
            fill: layer.style?.fill,
            radius: layer.style?.radius,
          },
          ...mapped,
        });
      });
      return pending;
    },
    [effectiveLayoutConfig, generatedPrompt, mediaSlots, uiState],
  );

  const syncPreviewComposition = useCallback(
    async (reason: 'auto' | 'manual' = 'auto') => {
      if (!effectiveLayoutConfig) return null;
      const requestId = previewRequestRef.current + 1;
      previewRequestRef.current = requestId;
      setIsPreviewSyncing(true);
      setPreviewError('');

      const materials = mediaSlots
        .filter((slot) => slot.layerType !== 'background')
        .map((slot) => ({
          id: slot.id,
          name: slot.label,
          layer_type: slot.layerType,
          image_base64: slot.uri ? extractBase64(slot.uri) : undefined,
        }));

      try {
        const backgroundSize = resolveSizeForSlot(true, aspectRatio);
        const response = await api.post('/scene/preview', {
          aspect_ratio: aspectRatio,
          materials,
          text_layers: textLayers,
          text_panel_fill: 'rgba(255,255,255,0.78)',
        });

        const nextComposition: PreviewComposition = {
          layers: Array.isArray(response.data?.layers) ? response.data.layers : [],
          canvasWidth: Number(response.data?.canvas_width) || backgroundSize.width,
          canvasHeight: Number(response.data?.canvas_height) || backgroundSize.height,
          updatedAt: Date.now(),
        };

        if (previewRequestRef.current !== requestId) return null;
        setPreviewComposition(nextComposition);
        setPreviewItems(buildPendingItemsFromComposition(nextComposition));
        if (reason === 'manual') {
          setStatus({ type: 'success', message: '预览已同步，可进入画布。' });
        }
        return nextComposition;
      } catch (error) {
        if (previewRequestRef.current === requestId) {
          setPreviewError(toApiErrorMessage(error, '实时预览同步失败。'));
          if (reason === 'manual') {
            setStatus({ type: 'error', message: toApiErrorMessage(error, '生成预览失败。') });
          }
        }
        return null;
      } finally {
        if (previewRequestRef.current === requestId) {
          setIsPreviewSyncing(false);
        }
      }
    },
    [aspectRatio, buildPendingItemsFromComposition, effectiveLayoutConfig, mediaSlots, textLayers],
  );

  useEffect(() => {
    if (!effectiveLayoutConfig) {
      setPreviewComposition(null);
      setPreviewItems([]);
      setPreviewError('');
      setIsPreviewSyncing(false);
      return;
    }
    const timer = window.setTimeout(() => {
      void syncPreviewComposition('auto');
    }, 350);
    return () => window.clearTimeout(timer);
  }, [effectiveLayoutConfig, mediaSlots, textLayers, aspectRatio, syncPreviewComposition]);

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
      const planRes = await api.post('/scene/plan', {
        prompt: initialPrompt,
        chat_history: chatHistory,
        selected_options: selectedOptions,
      });

      const layoutPayload: LayoutPayload = planRes.data;
      const nextLayout = layoutPayload.layout_config ?? layoutPayload.layoutConfig;
      const nextDraft = normalizeSceneDraft(layoutPayload.draft, initialPrompt, selectedOptions);
      const nextLayerPlan = layoutPayload.layer_plan ?? layoutPayload.layerPlan ?? null;
      setLayoutConfig(nextLayout);
      setLayerPlan(nextLayerPlan);
      setSceneDraft(nextDraft);
      setGeneratedPrompt(
        nextDraft.controls.positive_prompt ??
          layoutPayload.draft?.controls?.positivePrompt ??
          layoutPayload.text_response ??
          layoutPayload.textResponse ??
          '',
      );
      setUiState(buildSceneUiState(ensureStableEditSections(nextLayout, nextLayerPlan, nextDraft), nextDraft));
      setPreviewComposition(null);
      setPreviewItems([]);
      setPreviewError('');
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
      updateSlot(componentId, slotId, (slot) => ({
        ...slot,
        uri: dataUrl,
        hasTransparentBg: slot.layerType === 'background' ? false : detectAlphaFromDataUrl(dataUrl),
      }));
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
        const imgRes = await api.post('/scene/materials/generate', {
          prompt: slot.prompt,
          width: size.width,
          height: size.height,
          layer_type: slot.layerType,
          transparent_background: !isBackground,
        });
        const base64 = imgRes.data?.image_base64;
        if (!base64) throw new Error('未返回图片数据');
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        updateSlot(componentId, slot.id, (prev) => ({
          ...prev,
          uri: dataUrl,
          hasTransparentBg: Boolean(imgRes.data?.transparent_background || !isBackground),
        }));
        setStatus({
          type: 'success',
          message: resolveImageGenerationStatusMessage(imgRes.data, '素材生成完成。'),
        });
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
    if (!effectiveLayoutConfig) {
      setStatus({ type: 'error', message: '请先生成编辑界面。' });
      return;
    }
    const ownerMap = new Map<string, string>();
    (effectiveLayoutConfig.sections ?? []).forEach((section) => {
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
    let fallbackCount = 0;
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
        const imgRes = await api.post('/scene/materials/generate', {
          prompt: slot.prompt,
          width: size.width,
          height: size.height,
          layer_type: slot.layerType,
          transparent_background: !isBackground,
        });
        const base64 = imgRes.data?.image_base64;
        if (!base64) {
          throw new Error('未返回图片数据');
        }
        if (imgRes.data?.fallback_used) {
          fallbackCount += 1;
        }
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        updateSlot(componentId, slot.id, (prev) => ({
          ...prev,
          uri: dataUrl,
          hasTransparentBg: Boolean(imgRes.data?.transparent_background || !isBackground),
        }));
      }
      setStatus({
        type: 'success',
        message:
          fallbackCount > 0
            ? `批量生成完成（${slotsToGenerate.length} 个，其中 ${fallbackCount} 个使用占位图）。`
            : `批量生成完成（${slotsToGenerate.length} 个）。`,
      });
    } catch (error) {
      const msg = toApiErrorMessage(error, '批量生成失败。');
      setStatus({ type: 'error', message: msg });
      window.alert(msg);
    } finally {
      setIsWorking(false);
    }
  }, [aspectRatio, effectiveLayoutConfig, mediaSlots, uiState, updateSlot]);

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

  const handleEnterCanvas = useCallback(async () => {
    setIsWorking(true);
    setStatus({ type: 'loading', message: '正在同步当前预览并进入画布...' });
    try {
      const latest = await syncPreviewComposition('manual');
      const items = latest ? buildPendingItemsFromComposition(latest) : previewItems;
      if (!items.length) {
        throw new Error('当前没有可导入画布的预览结果。');
      }
      sessionStorage.setItem('pendingCanvasItems', JSON.stringify(items));
      navigate('/canvas');
    } catch (error) {
      const msg = toApiErrorMessage(error, '进入画布失败。');
      setStatus({ type: 'error', message: msg });
      window.alert(msg);
    } finally {
      setIsWorking(false);
    }
  }, [buildPendingItemsFromComposition, navigate, previewItems, syncPreviewComposition]);

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
      setQuickStatus({
        type: 'success',
        message: resolveImageGenerationStatusMessage(res.data, '快速图像已生成。'),
      });
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

  const layoutSections = useMemo(
    () => (effectiveLayoutConfig?.sections ?? []).filter((section) => !isReservedSection(section)),
    [effectiveLayoutConfig],
  );
  const materialsSection = useMemo(
    () => layoutSections.find((section) => (section.components ?? []).some((component) => component.type === 'media-uploader')),
    [layoutSections],
  );
  const configSections = useMemo(
    () => layoutSections.filter((section) => section.id !== materialsSection?.id),
    [layoutSections, materialsSection?.id],
  );
  const layouts = useMemo<Layouts>(() => {
    const buildForCols = (cols: number) => {
      const items: LayoutItemSpec[] = [];
      const isSingleColumn = cols <= GRID_COLS.sm;
      const step1Height = Math.max(10, step1Rows || 12);

      if (isSingleColumn) {
        items.push({ i: 'chat', w: cols, h: step1Height });
        items.push({ i: 'form', w: cols, h: step1Height });
      } else {
        const chatWidth = Math.max(1, Math.round((cols * 2) / 3));
        const formWidth = Math.max(1, cols - chatWidth);
        items.push({ i: 'chat', w: chatWidth, h: step1Height });
        items.push({ i: 'form', w: formWidth, h: step1Height });
      }

      if (showQuick) {
        const quickWidth = isSingleColumn ? cols : Math.max(1, cols - Math.max(1, Math.round((cols * 2) / 3)));
        items.push({ i: 'quick', w: quickWidth, h: 7 });
      }

      return packGridItems(items, cols);
    };
    return {
      lg: buildForCols(GRID_COLS.lg),
      md: buildForCols(GRID_COLS.md),
      sm: buildForCols(GRID_COLS.sm),
      xs: buildForCols(GRID_COLS.xs),
    };
  }, [showQuick, step1Rows]);

  const previewCanvasWidth = previewComposition?.canvasWidth || resolveSizeForSlot(true, aspectRatio).width;
  const previewCanvasHeight = previewComposition?.canvasHeight || resolveSizeForSlot(true, aspectRatio).height;
  if (effectiveLayoutConfig) {
    return (
      <SceneEditWorkspace
        activeStep={activeStep}
        flowSteps={FLOW_STEPS}
        status={status}
        isWorking={isWorking}
        isPreviewSyncing={isPreviewSyncing}
        previewError={previewError}
        previewComposition={previewComposition}
        previewItemsCount={previewItems.length}
        previewCanvasWidth={previewCanvasWidth}
        previewCanvasHeight={previewCanvasHeight}
        layerPlan={layerPlan}
        sceneDraft={sceneDraft}
        materialsSection={materialsSection}
        configSections={configSections}
        uiState={uiState}
        mediaSlots={mediaSlots}
        textLayers={textLayers}
        onStateChange={handleStateChange}
        onSlotUpload={handleSlotUpload}
        onSlotGenerate={handleSlotGenerate}
        onSlotRemoveBackground={handleSlotRemoveBackground}
        onGenerateAllSlots={handleGenerateAllSlots}
        onEnterCanvas={handleEnterCanvas}
        enterCanvasDisabled={(isSending || isWorking) || isPreviewSyncing || !previewItems.length}
      />
    );
  }

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
      {/* <Typography variant="h4" component="h1" gutterBottom>
        AI 场景生成
      </Typography> */}

      <Stepper activeStep={activeStep} sx={{ mb: 1 }}>
        {FLOW_STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box ref={gridContainerRef} sx={{ flex: 1, minHeight: 0, overflow: 'auto', pb: 12, pr: 1 }}>
        <ResponsiveGridLayout
          className="scene-dashboard-grid"
          layouts={layouts}
          breakpoints={GRID_BREAKPOINTS}
          cols={GRID_COLS}
          rowHeight={GRID_ROW_HEIGHT}
          margin={GRID_MARGIN}
          containerPadding={GRID_CONTAINER_PADDING}
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
          <Stack direction="row" spacing={1} alignItems="center">
            <Button variant="contained" onClick={handleGenerateEdit} disabled={isSending}>
              生成编辑界面
            </Button>
            <Button variant="text" onClick={() => setShowQuick((prev) => !prev)} disabled={isSending}>
              {showQuick ? '隐藏快速模式' : '快速模式'}
            </Button>
          </Stack>
        </Paper>
      </Box>

      <Backdrop open={isWorking} sx={{ zIndex: (theme) => theme.zIndex.drawer + 10, color: '#fff' }}>
        <Stack spacing={2} alignItems="center">
          <CircularProgress color="inherit" />
          <Typography variant="body2">{status.message || '处理中...'}</Typography>
        </Stack>
      </Backdrop>
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

function isReservedSection(section?: LayoutSection): boolean {
  if (!section?.cardType) return false;
  const type = section.cardType.toLowerCase();
  return type === 'preview' || type === 'intent-board' || type === 'intent' || type === 'board';
}

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

function ensureStableEditSections(
  layoutConfig: LayoutConfig | undefined,
  layerPlan: any,
  sceneDraft?: SceneDraft,
): LayoutConfig | undefined {
  if (!layoutConfig) return layoutConfig;
  const sections = Array.isArray(layoutConfig.sections) ? layoutConfig.sections : [];
  const slots = buildMaterialsSlotsFromDraft(sceneDraft).length
    ? buildMaterialsSlotsFromDraft(sceneDraft)
    : buildMaterialsSlotsFromLayerPlan(layerPlan);
  let repaired = false;
  const nextSections = sections.map((section) => {
    let changed = false;
    const nextComponents = (section.components ?? []).map((component: any) => {
      if (component.type !== 'media-uploader') return component;
      if (!slots.length) return component;
      const mergedSlots = mergeMediaSlots(component.slots, slots);
      const hasSameSlots =
        Array.isArray(component.slots) &&
        component.slots.length === mergedSlots.length &&
        component.slots.every((slot: MediaSlot, index: number) => isSameSlot(slot, mergedSlots[index]));
      if (hasSameSlots) return component;
      changed = true;
      repaired = true;
      return {
        ...component,
        slots: mergedSlots,
      };
    });
    if (!changed) return section;
    return {
      ...section,
      components: nextComponents,
    };
  });
  const hasMaterials = nextSections.some((section) =>
    (section.components ?? []).some((component: any) => component.type === 'media-uploader'),
  );
  if (hasMaterials) {
    const withFallbacks = appendStableFallbackSections(nextSections, sceneDraft, slots);
    const structureChanged =
      withFallbacks.length !== sections.length ||
      withFallbacks.some((section, index) => section !== sections[index]);
    return repaired || structureChanged ? { ...layoutConfig, sections: withFallbacks } : layoutConfig;
  }

  return {
    ...layoutConfig,
    sections: appendStableFallbackSections(nextSections, sceneDraft, slots),
  };
}

function buildMaterialsSlotsFromLayerPlan(layerPlan: any): MediaSlot[] {
  const planLayers = Array.isArray(layerPlan?.layer_plan) ? layerPlan.layer_plan : [];
  return planLayers
    .filter((layer: any) => {
      const type = String(layer?.layer_type || '').toLowerCase();
      return type === 'background' || type === 'subject' || type === 'decor';
    })
    .map((layer: any, index: number) => ({
      id: String(layer?.layer_id || layer?.id || `slot-${index + 1}`),
      label: String(layer?.layer_name || layer?.name || `图层 ${index + 1}`),
      layerType: normalizeLayerType(layer?.layer_type),
      prompt: String(layer?.prompt || '').trim() || `生成${String(layer?.layer_name || layer?.name || `图层 ${index + 1}`)}`,
    }));
}

function buildMaterialsSlotsFromDraft(sceneDraft?: SceneDraft): MediaSlot[] {
  const draftSlots = Array.isArray(sceneDraft?.materials?.slots) ? sceneDraft.materials.slots : [];
  return draftSlots
    .map((slot, index) => normalizeDraftMaterialSlot(slot, index))
    .filter(Boolean) as MediaSlot[];
}

function normalizeDraftMaterialSlot(slot: SceneMaterialDraftSlot | null | undefined, index = 0): MediaSlot | null {
  if (!slot || typeof slot !== 'object') return null;
  const label = String(slot.label || '').trim() || `素材 ${index + 1}`;
  const prompt = String(slot.prompt || '').trim() || `生成${label}`;
  return {
    id: String(slot.id || `draft-slot-${index + 1}`),
    label,
    layerType: normalizeLayerType(slot.layer_type),
    prompt,
  };
}

function normalizeLayerType(value: unknown): 'background' | 'subject' | 'decor' {
  const type = String(value || '').trim().toLowerCase();
  if (type === 'background' || type === 'subject' || type === 'decor') {
    return type;
  }
  return 'subject';
}

function mergeMediaSlots(existingSlots: unknown, draftSlots: MediaSlot[]): MediaSlot[] {
  const current = Array.isArray(existingSlots) ? existingSlots : [];
  if (!current.length) {
    return draftSlots.map((slot) => ({ ...slot }));
  }
  if (!draftSlots.length) {
    return current.map((slot) => ({ ...slot }));
  }

  const next = current.map((slot: MediaSlot) => {
    const match = findMatchingSlot(slot, draftSlots);
    if (!match) return { ...slot };
    return {
      ...slot,
      label: match.label || slot.label,
      layerType: match.layerType || slot.layerType,
      prompt: match.prompt || slot.prompt,
      uri: slot.uri,
      hasTransparentBg: slot.hasTransparentBg,
    };
  });

  draftSlots.forEach((slot) => {
    const exists = next.some((candidate) => isSlotMatch(candidate, slot));
    if (!exists) {
      next.push({ ...slot });
    }
  });

  return next;
}

function findMatchingSlot(slot: MediaSlot, candidates: MediaSlot[]): MediaSlot | undefined {
  return candidates.find((candidate) => isSlotMatch(slot, candidate));
}

function isSlotMatch(left: MediaSlot, right: MediaSlot): boolean {
  return (
    String(left.id || '').trim() === String(right.id || '').trim() ||
    (
      normalizeLayerType(left.layerType) === normalizeLayerType(right.layerType) &&
      String(left.label || '').trim() === String(right.label || '').trim()
    )
  );
}

function isSameSlot(left: MediaSlot | undefined, right: MediaSlot | undefined): boolean {
  if (!left || !right) return false;
  return (
    left.id === right.id &&
    left.label === right.label &&
    left.layerType === right.layerType &&
    left.prompt === right.prompt &&
    left.uri === right.uri &&
    left.hasTransparentBg === right.hasTransparentBg
  );
}

function appendStableFallbackSections(
  sections: LayoutSection[],
  sceneDraft: SceneDraft | undefined,
  slots: MediaSlot[],
): LayoutSection[] {
  const nextSections = [...sections];
  const hasMaterials = nextSections.some((section) =>
    (section.components ?? []).some((component: any) => component.type === 'media-uploader'),
  );
  const hasCopy = nextSections.some((section) => isCopySection(section));
  const hasControls = nextSections.some((section) => isControlSection(section));

  if (!hasMaterials && slots.length) {
    nextSections.unshift(buildMaterialsFallbackSection(slots));
  }
  if (!hasCopy) {
    nextSections.push(buildCopyFallbackSection(sceneDraft));
  }
  if (!hasControls) {
    nextSections.push(buildControlsFallbackSection(sceneDraft));
  }
  return dedupeSections(nextSections);
}

function dedupeSections(sections: LayoutSection[]): LayoutSection[] {
  const seen = new Set<string>();
  return sections.filter((section) => {
    const id = String(section.id || '').trim();
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

function buildMaterialsFallbackSection(slots: MediaSlot[]): LayoutSection {
  return {
    id: 'materials-fallback',
    cardType: 'materials',
    title: '素材准备',
    description: '当前布局未返回素材区，已按场景规划自动补齐',
    layout: { span: 2, tone: 'soft' },
    components: [
      {
        id: 'materials-uploader-fallback',
        type: 'media-uploader',
        title: '上传或生成素材',
        slots: slots.map((slot) => ({ ...slot })),
      },
    ],
  };
}

function buildCopyFallbackSection(sceneDraft?: SceneDraft): LayoutSection {
  return {
    id: 'copywriting-fallback',
    cardType: 'copywriting',
    title: '文案与内容',
    description: '文案区缺失时使用稳定的编辑骨架兜底',
    layout: { span: 2, tone: 'soft' },
    components: [
      {
        id: 'headline',
        type: 'text-input',
        label: '主标题',
        placeholder: '输入主标题',
        default: sceneDraft?.copy?.headline || '',
      },
      {
        id: 'subtitle',
        type: 'text-input',
        label: '副标题',
        placeholder: '输入副标题',
        default: sceneDraft?.copy?.subtitle || '',
      },
      {
        id: 'body-copy',
        type: 'textarea',
        label: '正文文案',
        placeholder: '输入补充文案',
        default: sceneDraft?.copy?.body || '',
      },
      {
        id: 'primary-color',
        type: 'color-palette',
        label: '主色调',
        default: sceneDraft?.copy?.primary_color || '#F4A261',
        options: [
          { value: '#F4A261', label: '暖橙', color: '#F4A261' },
          { value: '#264653', label: '深蓝', color: '#264653' },
          { value: '#111827', label: '深灰', color: '#111827' },
          { value: '#2A9D8F', label: '青绿', color: '#2A9D8F' },
          { value: '#E63946', label: '朱红', color: '#E63946' },
        ],
        helperText: '文案和形状层会优先使用这个颜色',
      },
    ],
  };
}

function buildControlsFallbackSection(sceneDraft?: SceneDraft): LayoutSection {
  return {
    id: 'controls-fallback',
    cardType: 'controls',
    title: '生成控制',
    description: '提示词和参数区缺失时使用稳定兜底',
    layout: { span: 2, tone: 'soft' },
    components: [
      {
        id: 'aspect-ratio',
        type: 'ratio-select',
        label: '画幅比例',
        default: sceneDraft?.brief?.aspect_ratio || '3:4',
        options: ['1:1', '3:4', '4:3', '9:16', '16:9'],
      },
      {
        id: 'generation-prompts',
        type: 'prompt-editor',
        title: '提示词',
        fields: [
          {
            id: 'positive',
            label: '正向提示词',
            default: sceneDraft?.controls?.positive_prompt || sceneDraft?.brief?.prompt || '',
          },
          {
            id: 'negative',
            label: '负向提示词',
            default: sceneDraft?.controls?.negative_prompt || '',
          },
        ],
      },
      {
        id: 'steps',
        type: 'slider',
        label: '采样步数',
        min: 10,
        max: 60,
        step: 1,
        default: sceneDraft?.controls?.steps ?? 28,
      },
      {
        id: 'cfg-scale',
        type: 'slider',
        label: 'CFG',
        min: 1,
        max: 15,
        step: 0.5,
        default: sceneDraft?.controls?.cfg_scale ?? 7,
      },
      {
        id: 'seed-lock',
        type: 'toggle',
        label: '锁定种子',
        default: Boolean(sceneDraft?.controls?.seed_locked),
        helperText: '锁定后重复生成会更稳定',
      },
    ],
  };
}

function buildSceneUiState(layoutConfig?: LayoutConfig, sceneDraft?: SceneDraft): Record<string, any> {
  const next = initializeUiState(layoutConfig);
  if (!layoutConfig?.sections?.length || !sceneDraft) return next;

  layoutConfig.sections.forEach((section) => {
    (section.components ?? []).forEach((component: any) => {
      if (component.type === 'media-uploader') {
        next[component.id] = mergeMediaSlots(next[component.id], buildMaterialsSlotsFromDraft(sceneDraft));
        return;
      }
      if (component.type === 'prompt-editor') {
        const current = typeof next[component.id] === 'object' && next[component.id] ? next[component.id] : {};
        next[component.id] = {
          ...current,
          positive: sceneDraft.controls.positive_prompt || current.positive || '',
          negative: sceneDraft.controls.negative_prompt || current.negative || '',
        };
        return;
      }
      if (matchesCopyField(component, 'headline')) {
        next[component.id] = sceneDraft.copy.headline || next[component.id] || '';
        return;
      }
      if (matchesCopyField(component, 'subtitle')) {
        next[component.id] = sceneDraft.copy.subtitle || next[component.id] || '';
        return;
      }
      if (matchesCopyField(component, 'body')) {
        next[component.id] = sceneDraft.copy.body || next[component.id] || '';
        return;
      }
      if (component.type === 'color-palette' && isPrimaryColorComponent(component)) {
        next[component.id] = resolveColorPaletteValue(component, sceneDraft.copy.primary_color || '#F4A261');
        return;
      }
      if (isAspectRatioComponent(component)) {
        next[component.id] = sceneDraft.brief.aspect_ratio || next[component.id] || '3:4';
        return;
      }
      if (isStepsComponent(component)) {
        next[component.id] = sceneDraft.controls.steps ?? next[component.id] ?? 28;
        return;
      }
      if (isCfgScaleComponent(component)) {
        next[component.id] = sceneDraft.controls.cfg_scale ?? next[component.id] ?? 7;
        return;
      }
      if (isSeedLockComponent(component)) {
        next[component.id] = Boolean(sceneDraft.controls.seed_locked);
      }
    });
  });

  return next;
}

function normalizeSceneDraft(
  value: Partial<SceneDraft> | undefined,
  prompt: string,
  selectedOptions: Record<string, string[]>,
): SceneDraft {
  const normalizedSlots = normalizeDraftMaterials(value?.materials?.slots, value?.materials);
  const background = normalizedSlots.find((slot) => slot.layer_type === 'background') ?? null;
  const subjects = normalizedSlots.filter((slot) => slot.layer_type === 'subject');
  const decors = normalizedSlots.filter((slot) => slot.layer_type === 'decor');

  return {
    brief: {
      prompt: safeText(value?.brief?.prompt, prompt),
      summary: safeText(value?.brief?.summary, prompt),
      aspect_ratio: safeText(value?.brief?.aspect_ratio, selectedOptions['画幅比例']?.[0] || '3:4'),
    },
    copy: {
      headline: safeText(value?.copy?.headline),
      subtitle: safeText(value?.copy?.subtitle),
      body: safeText(value?.copy?.body),
      primary_color: safeText(value?.copy?.primary_color, '#F4A261'),
    },
    controls: {
      positive_prompt: safeText(
        value?.controls?.positive_prompt ?? value?.controls?.positivePrompt,
        prompt,
      ),
      negative_prompt: safeText(value?.controls?.negative_prompt),
      steps: safeNumber(value?.controls?.steps, 28),
      cfg_scale: safeNumber(value?.controls?.cfg_scale, 7),
      seed_locked: Boolean(value?.controls?.seed_locked),
    },
    materials: {
      background,
      subjects,
      decors,
      slots: normalizedSlots,
    },
  };
}

function normalizeDraftMaterials(
  slotsValue: SceneDraft['materials']['slots'] | undefined,
  materialsValue: Partial<SceneDraft['materials']> | undefined,
): SceneDraft['materials']['slots'] {
  const directSlots = Array.isArray(slotsValue) ? slotsValue : [];
  const grouped = [
    materialsValue?.background,
    ...(Array.isArray(materialsValue?.subjects) ? materialsValue.subjects : []),
    ...(Array.isArray(materialsValue?.decors) ? materialsValue.decors : []),
  ];
  const source = directSlots.length ? directSlots : grouped;

  return source
    .map((slot, index) => {
      if (!slot || typeof slot !== 'object') return null;
      return {
        id: safeText((slot as SceneMaterialDraftSlot).id, `draft-slot-${index + 1}`),
        label: safeText((slot as SceneMaterialDraftSlot).label, `素材 ${index + 1}`),
        layer_type: normalizeLayerType((slot as SceneMaterialDraftSlot).layer_type),
        prompt: safeText((slot as SceneMaterialDraftSlot).prompt, `生成素材 ${index + 1}`),
      };
    })
    .filter(Boolean) as SceneDraft['materials']['slots'];
}

function safeText(value: unknown, fallback = ''): string {
  const text = typeof value === 'string' ? value.trim() : '';
  return text || fallback;
}

function safeNumber(value: unknown, fallback: number): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function matchesCopyField(component: any, kind: 'headline' | 'subtitle' | 'body'): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  if (kind === 'headline') return /headline|title|主标题|标题|scene-title|场景名称/.test(sample);
  if (kind === 'subtitle') return /subtitle|副标题|slogan|标语|卖点/.test(sample);
  return /body|copy|正文|文案/.test(sample);
}

function isPrimaryColorComponent(component: any): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  return /color|palette|主色|色调/.test(sample);
}

function resolveColorPaletteValue(component: any, desiredColor: string): string {
  const options = Array.isArray(component?.options) ? component.options : [];
  const normalizedColor = String(desiredColor || '').trim().toLowerCase();
  const matched = options.find((option: any) => {
    if (typeof option === 'string') {
      return option.trim().toLowerCase() === normalizedColor;
    }
    return (
      String(option?.value || '').trim().toLowerCase() === normalizedColor ||
      String(option?.color || '').trim().toLowerCase() === normalizedColor
    );
  });
  if (typeof matched === 'string') return matched;
  if (matched && typeof matched === 'object') {
    return String(matched.value || matched.color || desiredColor);
  }
  return desiredColor;
}

function isAspectRatioComponent(component: any): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  return component?.type === 'ratio-select' || /ratio|aspect|比例|画幅/.test(sample);
}

function isStepsComponent(component: any): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  return /steps|采样步数/.test(sample);
}

function isCfgScaleComponent(component: any): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  return /cfg|guidance/.test(sample);
}

function isSeedLockComponent(component: any): boolean {
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  return /seed-lock|seed|种子/.test(sample);
}

function isCopySection(section: LayoutSection): boolean {
  if ((section.cardType || '').toLowerCase() === 'copywriting') return true;
  return (section.components ?? []).some((component: any) => {
    const sample = `${component.id || ''} ${component.label || ''} ${component.title || ''}`.toLowerCase();
    return /标题|headline|subtitle|文案|copy|body/.test(sample);
  });
}

function isControlSection(section: LayoutSection): boolean {
  return (section.components ?? []).some((component: any) => {
    const sample = `${component.id || ''} ${component.label || ''} ${component.title || ''}`.toLowerCase();
    return component.type === 'prompt-editor' || /steps|cfg|seed|ratio|画幅|提示词/.test(sample);
  });
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
      } else if (component.type === 'color-palette') {
        const allowMultiple = 'allowMultiple' in component && Boolean((component as any).allowMultiple);
        next[component.id] =
          allowMultiple && Array.isArray(component.default)
            ? component.default
            : component.default ?? '';
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

function collectTextLayers(
  layoutConfig: LayoutConfig | undefined,
  uiState: Record<string, any>,
  selectedOptions: Record<string, string[]>,
): TextLayerDraft[] {
  if (!layoutConfig?.sections?.length) return [];
  const results: TextLayerDraft[] = [];
  const color = findPrimaryColor(layoutConfig, uiState) || '#111827';

  layoutConfig.sections.forEach((section) => {
    (section.components ?? []).forEach((component: any) => {
      const label = component.label || component.title || component.id;
      if ((component.type === 'text-input' || component.type === 'textarea') && shouldExportCanvasText(component.id, label)) {
        const text = normalizeText(uiState[component.id]);
        if (!text) return;
        results.push({
          id: component.id,
          name: label,
          text,
          role: inferTextRole(label, text),
          color,
          align: 'left',
        });
      }
    });
  });

  if (!results.length) {
    const mood = selectedOptions['氛围']?.[0];
    if (mood) {
      results.push({
        id: 'fallback-subtitle',
        name: '副标题',
        text: mood,
        role: 'subtitle',
        color,
        align: 'left',
      });
    }
  }

  const unique = new Map<string, TextLayerDraft>();
  results.forEach((item) => {
    const key = `${item.role}:${item.text}`;
    if (!unique.has(key)) unique.set(key, item);
  });
  return Array.from(unique.values()).slice(0, 4);
}

function findPrimaryColor(layoutConfig: LayoutConfig | undefined, uiState: Record<string, any>): string | undefined {
  if (!layoutConfig?.sections?.length) return undefined;
  for (const section of layoutConfig.sections) {
    for (const component of section.components ?? []) {
      if ((component as any).type !== 'color-palette') continue;
      const paletteOptions = Array.isArray((component as any).options) ? (component as any).options : [];
      const resolvePaletteColor = (entry: unknown): string | undefined => {
        if (typeof entry !== 'string' || !entry.trim()) return undefined;
        const normalized = entry.trim();
        const matched = paletteOptions.find((option: any) => {
          if (typeof option === 'string') {
            return option.trim() === normalized;
          }
          return String(option?.value ?? '').trim() === normalized;
        });
        if (matched && typeof matched === 'object') {
          const swatch = typeof matched.color === 'string' ? matched.color.trim() : '';
          if (swatch) return swatch;
        }
        return /^#|^rgb|^hsl/i.test(normalized) ? normalized : undefined;
      };
      const value = uiState[(component as any).id];
      if (Array.isArray(value)) {
        for (const entry of value) {
          const color = resolvePaletteColor(entry);
          if (color) return color;
        }
        continue;
      }
      const color = resolvePaletteColor(value);
      if (color) return color;
    }
  }
  return undefined;
}

function inferTextRole(label: string, text: string): 'title' | 'subtitle' | 'body' {
  const sample = `${label} ${text}`.toLowerCase();
  if (/标题|title|headline|主标题|场景名称|scene-title/.test(sample)) return 'title';
  if (/副标题|subtitle|slogan|标语|卖点/.test(sample)) return 'subtitle';
  return text.length <= 18 ? 'subtitle' : 'body';
}

function shouldExportCanvasText(id: string, label: string): boolean {
  const sample = `${id} ${label}`.toLowerCase();
  if (/prompt|negative|scene-notes|提示词/.test(sample)) {
    return false;
  }
  if (/备注|说明|补充描述/.test(sample)) {
    return false;
  }
  return /标题|title|headline|副标题|subtitle|slogan|标语|文案|卖点|正文|body|copy|场景名称|scene-title/.test(sample);
}

function normalizeText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
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
  if (!isBackground) return { width: 1024, height: 1024 };
  const ratio = parseAspectRatio(aspectRatio);
  const width = 1536;
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

function resolveImageGenerationStatusMessage(data: any, successMessage: string) {
  if (!data?.fallback_used) {
    return successMessage;
  }
  const reason =
    typeof data?.fallback_reason === 'string' && data.fallback_reason.trim()
      ? ` 原因：${data.fallback_reason.trim()}`
      : '';
  return `模型不可用，已自动使用占位图继续流程。${reason}`;
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

function detectAlphaFromDataUrl(dataUrl: string): boolean {
  return dataUrl.startsWith('data:image/png') || dataUrl.startsWith('data:image/webp');
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
