import React, { useCallback, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Divider,
  Grid,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import axios from 'axios';
import { useLocation, useNavigate } from 'react-router-dom';
import { api } from '../../api';
import DynamicRenderer, { type MediaSlot } from './dynamic/DynamicRenderer';

type LocationState = {
  layoutConfig: any;
  generatedPrompt?: string;
  chatHistory?: string;
  selectedOptions?: Record<string, string[]>;
  layerPlan?: any;
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

const CANVAS_DIMENSION = 4000;
const TARGET_BG_WIDTH = 2400;

const SceneEditPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state || {}) as LocationState;
  const layoutConfig = state.layoutConfig as LayoutConfig | undefined;

  const [isWorking, setIsWorking] = useState(false);
  const [uiState, setUiState] = useState<Record<string, any>>(() =>
    initializeUiState(layoutConfig),
  );

  const selectedOptions = state.selectedOptions || {};

  const aspectRatio = useMemo(() => {
    const ratioField = layoutConfig?.meta?.aspect_ratio_field;
    const fromUi = ratioField ? uiState?.[ratioField] : undefined;
    const fromSelected = selectedOptions['画幅比例']?.[0];
    const fromMeta = layoutConfig?.meta?.aspect_ratio;
    return fromUi || fromSelected || fromMeta || '3:4';
  }, [layoutConfig?.meta?.aspect_ratio, layoutConfig?.meta?.aspect_ratio_field, selectedOptions, uiState]);

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
      } catch (error) {
        const msg = toApiErrorMessage(error, '生成素材失败。');
        window.alert(msg);
      } finally {
        setIsWorking(false);
      }
    },
    [aspectRatio, updateSlot],
  );

  const handleSlotRemoveBackground = useCallback(
    async (componentId: string, slot: MediaSlot) => {
      if (!slot.uri) return;
      setIsWorking(true);
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
      } catch (error) {
        const msg = toApiErrorMessage(error, '抠图失败。');
        window.alert(msg);
      } finally {
        setIsWorking(false);
      }
    },
    [updateSlot],
  );

  const handleGeneratePreviewToCanvas = useCallback(async () => {
    setIsWorking(true);
    try {
      const mediaSlots = extractMediaSlots(layoutConfig, uiState);
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
          buildPlaceholderDataUrl(`background:${state.generatedPrompt || ''}`, bgW, bgH);
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
      navigate('/canvas');
    } catch (error) {
      const msg = toApiErrorMessage(error, '生成预览失败。');
      window.alert(msg);
    } finally {
      setIsWorking(false);
    }
  }, [aspectRatio, layoutConfig, navigate, state.generatedPrompt, uiState]);

  if (!layoutConfig) {
    return (
      <Container maxWidth={false} sx={{ mt: 4 }}>
        <Typography variant="h5">缺少编辑上下文</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          请从「AI 场景生成 · Chat」开始。
        </Typography>
        <Button sx={{ mt: 2 }} variant="contained" onClick={() => navigate('/scene/chat')}>
          返回 Chat
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth={false} sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        AI 场景生成 · Edit（平板模式）
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                配置与素材
              </Typography>
              <Typography variant="body2" color="text.secondary">
                画幅：{aspectRatio}
              </Typography>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                选择题结果
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
                生成目标概要
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {layoutConfig.meta?.summary || '在左侧配置素材，右侧预览后落到画布。'}
              </Typography>
              {state.generatedPrompt ? (
                <TextField
                  value={state.generatedPrompt}
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

              <Button
                variant="contained"
                sx={{ mt: 3 }}
                onClick={handleGeneratePreviewToCanvas}
                disabled={isWorking}
                fullWidth
              >
                生成预览并进入画布
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                预览（落地到画布）
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                这里展示素材准备情况；点击左侧“生成预览并进入画布”后，会把背景与素材层导入 `CanvasPage`。
              </Typography>
              <Stack spacing={1.5}>
                {extractMediaSlots(layoutConfig, uiState).map((slot) => (
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
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                对话记录（只读）
              </Typography>
              <TextField
                value={state.chatHistory || ''}
                fullWidth
                multiline
                minRows={6}
                InputProps={{ readOnly: true }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default SceneEditPage;

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
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#f0f3f7"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#3c3c46" font-size="18" font-family="Arial">${escapeXml(label)}</text></svg>`;
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
