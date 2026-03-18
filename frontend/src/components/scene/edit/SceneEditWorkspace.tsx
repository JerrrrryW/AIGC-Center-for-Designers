import React, { useMemo } from 'react';
import {
  Alert,
  Backdrop,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  IconButton,
  LinearProgress,
  Stack,
  Step,
  StepLabel,
  Stepper,
  TextField,
  Typography,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { SectionCard, type LayoutSection, type MediaSlot } from '../dynamic/DynamicRenderer';
import { type SceneDraft } from './sceneDraft';

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

type SceneEditWorkspaceProps = {
  activeStep: number;
  flowSteps: string[];
  status: StatusState;
  isWorking: boolean;
  isPreviewSyncing: boolean;
  previewError: string;
  previewComposition: PreviewComposition | null;
  previewItemsCount: number;
  previewCanvasWidth: number;
  previewCanvasHeight: number;
  generatedPrompt: string;
  selectedOptions: Record<string, string[]>;
  layerPlan: any;
  sceneDraft?: SceneDraft;
  materialsSection?: LayoutSection;
  configSections: LayoutSection[];
  uiState: Record<string, any>;
  mediaSlots: MediaSlot[];
  textLayers: TextLayerDraft[];
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  onGenerateAllSlots: () => void;
  onEnterCanvas: () => void;
  enterCanvasDisabled: boolean;
};

const SceneEditWorkspace: React.FC<SceneEditWorkspaceProps> = ({
  activeStep,
  flowSteps,
  status,
  isWorking,
  isPreviewSyncing,
  previewError,
  previewComposition,
  previewItemsCount,
  previewCanvasWidth,
  previewCanvasHeight,
  generatedPrompt,
  selectedOptions,
  layerPlan,
  sceneDraft,
  materialsSection,
  configSections,
  uiState,
  mediaSlots,
  textLayers,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  onGenerateAllSlots,
  onEnterCanvas,
  enterCanvasDisabled,
}) => {
  const previewBackground = mediaSlots.find((slot) => slot.layerType === 'background')?.uri || null;
  const filledSlotCount = mediaSlots.filter((slot) => slot.uri).length;
  const copySections = useMemo(
    () => configSections.filter((section) => isCopySection(section)),
    [configSections],
  );
  const settingSections = useMemo(
    () => configSections.filter((section) => !isCopySection(section)),
    [configSections],
  );
  const previewSyncLabel = isPreviewSyncing ? '实时同步中' : previewComposition ? '实时预览已同步' : '等待预览';
  const copyReadyCount = [sceneDraft?.copy?.headline, sceneDraft?.copy?.subtitle, sceneDraft?.copy?.body].filter(Boolean).length;

  return (
    <Container
      maxWidth={false}
      sx={{
        mt: 3,
        mb: 2,
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        overflow: 'visible',
        position: 'relative',
        pb: 4,
      }}
    >
      <Stepper activeStep={activeStep} sx={{ mb: 1 }}>
        {flowSteps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box
        sx={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', xl: 'minmax(0, 1.15fr) minmax(360px, 0.85fr)' },
          gap: 2,
          alignItems: 'start',
          overflow: 'visible',
        }}
      >
        <Box sx={{ minWidth: 0, pr: { xl: 1 }, display: 'flex', flexDirection: 'column', gap: 2, overflow: 'visible' }}>
          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
            <Box>
              <Typography variant="h6">编辑工作台</Typography>
              <Typography variant="caption" color="text.secondary">
                固定素材区、文案区和参数区，避免布局波动影响编辑体验
              </Typography>
            </Box>
            <Chip label={previewSyncLabel} size="small" color={isPreviewSyncing ? 'default' : 'success'} />
          </Stack>

          {status.message ? (
            <Alert severity={status.type === 'error' ? 'error' : status.type === 'success' ? 'success' : 'info'}>
              {status.message}
            </Alert>
          ) : null}

          <SceneSetupCard
            materialsSection={materialsSection}
            uiState={uiState}
            sceneDraft={sceneDraft}
            generatedPrompt={generatedPrompt}
            selectedOptions={selectedOptions}
            layerPlan={layerPlan}
            filledSlotCount={filledSlotCount}
            mediaSlots={mediaSlots}
            copyReadyCount={copyReadyCount}
            onStateChange={onStateChange}
            onSlotUpload={onSlotUpload}
            onSlotGenerate={onSlotGenerate}
            onSlotRemoveBackground={onSlotRemoveBackground}
            onGenerateAllSlots={onGenerateAllSlots}
            disabled={isWorking}
          />

          {copySections.length ? (
            <>
              <SectionLabel
                title="文案与内容"
                description="标题、副标题和正文会直接进入实时预览与画布导出。"
              />
              {copySections.map((section) => (
                <SectionCard
                  key={section.id}
                  section={section}
                  cardType={section.cardType}
                  state={uiState}
                  onStateChange={onStateChange}
                  onSlotUpload={onSlotUpload}
                  onSlotGenerate={onSlotGenerate}
                  onSlotRemoveBackground={onSlotRemoveBackground}
                  disabled={isWorking}
                />
              ))}
            </>
          ) : null}

          {settingSections.length ? (
            <>
              <SectionLabel
                title="参数配置"
                description="风格、提示词和生成控制项会影响后续素材生成与预览合成。"
              />
              {settingSections.map((section) => (
                <SectionCard
                  key={section.id}
                  section={section}
                  cardType={section.cardType}
                  state={uiState}
                  onStateChange={onStateChange}
                  onSlotUpload={onSlotUpload}
                  onSlotGenerate={onSlotGenerate}
                  onSlotRemoveBackground={onSlotRemoveBackground}
                  disabled={isWorking}
                />
              ))}
            </>
          ) : null}
        </Box>

        <Box
          sx={{
            minWidth: 0,
            overflow: 'visible',
            position: { xs: 'static', xl: 'sticky' },
            top: { xl: 12 },
            alignSelf: 'start',
          }}
        >
          <Card
            sx={{
              height: { xs: 'auto', xl: 'calc(100vh - 140px)' },
              boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1, minHeight: 0 }}>
              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h6">实时预览</Typography>
                  <Typography variant="caption" color="text.secondary">
                    右侧预览与进入画布的数据保持一致
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} alignItems="center">
                  {previewError ? <Chip label="最近同步失败" size="small" color="warning" /> : null}
                  <Chip label={previewItemsCount ? `已生成 ${previewItemsCount} 层` : '占位预览'} size="small" />
                </Stack>
              </Stack>

              <Box
                sx={{
                  flex: 1,
                  minHeight: { xs: 360, xl: 0 },
                  borderRadius: 3,
                  overflow: 'hidden',
                  border: '1px solid',
                  borderColor: 'grey.200',
                  background: 'linear-gradient(180deg, rgba(248,250,252,1) 0%, rgba(238,242,247,1) 100%)',
                  position: 'relative',
                }}
              >
                <PreviewCanvas
                  width={previewCanvasWidth}
                  height={previewCanvasHeight}
                  backgroundUri={previewBackground}
                  layers={previewComposition?.layers || []}
                />
                {isPreviewSyncing ? (
                  <LinearProgress sx={{ position: 'absolute', left: 0, right: 0, top: 0 }} />
                ) : null}
              </Box>

              {textLayers.length ? (
                <Card variant="outlined">
                  <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="subtitle2">将导出到画布的文字层</Typography>
                    {textLayers.map((layer) => (
                      <Box key={layer.id}>
                        <Typography variant="caption" color="text.secondary">
                          {layer.name}
                        </Typography>
                        <Typography variant="body2">{layer.text}</Typography>
                      </Box>
                    ))}
                  </CardContent>
                </Card>
              ) : (
                <Alert severity="info">当前没有可导出到画布的文字层。只有标题/副标题/文案类字段会进入画布。</Alert>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  未点击“进入画布”前，右侧始终显示当前配置的最新结果；同步中会继续保留最近一次成功预览。
                </Typography>
                <Button variant="contained" onClick={onEnterCanvas} disabled={enterCanvasDisabled}>
                  进入画布
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Box>
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

const SectionLabel: React.FC<{
  title: string;
  description: string;
}> = ({ title, description }) => (
  <Box
    sx={{
      px: 1.5,
      py: 1.25,
      borderRadius: 2,
      border: '1px solid',
      borderColor: 'rgba(25, 118, 210, 0.18)',
      background: 'linear-gradient(180deg, rgba(239,246,255,0.92) 0%, rgba(255,255,255,0.96) 100%)',
    }}
  >
    <Typography variant="h6">{title}</Typography>
    <Typography variant="caption" color="text.secondary">
      {description}
    </Typography>
  </Box>
);

type SceneSetupCardProps = {
  materialsSection?: LayoutSection;
  uiState: Record<string, any>;
  sceneDraft?: SceneDraft;
  generatedPrompt: string;
  selectedOptions: Record<string, string[]>;
  layerPlan: any;
  filledSlotCount: number;
  mediaSlots: MediaSlot[];
  copyReadyCount: number;
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  onGenerateAllSlots: () => void;
  disabled: boolean;
};

const SceneSetupCard: React.FC<SceneSetupCardProps> = ({
  materialsSection,
  uiState,
  sceneDraft,
  generatedPrompt,
  selectedOptions,
  layerPlan,
  filledSlotCount,
  mediaSlots,
  copyReadyCount,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  onGenerateAllSlots,
  disabled,
}) => {
  const materialsComponent = (materialsSection?.components ?? []).find((component: any) => component.type === 'media-uploader');
  const componentId = materialsComponent?.id;
  const slots = componentId && Array.isArray(uiState[componentId]) ? uiState[componentId] : mediaSlots;

  return (
    <Card
      sx={{
        boxShadow: '0 18px 40px rgba(15, 23, 42, 0.12)',
        borderRadius: 3,
        border: '1px solid rgba(25, 118, 210, 0.14)',
        background: 'linear-gradient(180deg, rgba(248,252,255,0.98) 0%, rgba(255,255,255,1) 100%)',
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} alignItems={{ xs: 'flex-start', md: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="h6">素材准备与场景总览</Typography>
            <Typography variant="caption" color="text.secondary">
              先确认整体方向，再横向处理每个图层素材。
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
            <Chip label={`素材 ${filledSlotCount}/${mediaSlots.length}`} size="small" />
            <Button variant="outlined" size="small" onClick={onGenerateAllSlots} disabled={disabled}>
              一键生成所有图层
            </Button>
          </Stack>
        </Stack>

        <Typography variant="body2" color="text.secondary">
          {sceneDraft?.brief?.summary || '编辑信息已就绪，可以开始准备素材和文案。'}
        </Typography>

        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {sceneDraft?.brief?.aspect_ratio ? <Chip label={`画幅 ${sceneDraft.brief.aspect_ratio}`} size="small" /> : null}
          {sceneDraft?.materials?.slots?.length ? <Chip label={`规划素材 ${sceneDraft.materials.slots.length}`} size="small" /> : null}
          <Chip label={`文案字段 ${copyReadyCount}/3`} size="small" />
        </Stack>

        {generatedPrompt ? (
          <TextField
            value={generatedPrompt}
            fullWidth
            multiline
            minRows={2}
            label="系统提示（只读）"
            InputProps={{ readOnly: true }}
          />
        ) : null}

        {Object.keys(selectedOptions).length ? (
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {Object.entries(selectedOptions).flatMap(([key, values]) =>
              values.map((value) => (
                <Chip key={`${key}:${value}`} label={`${key}: ${value}`} size="small" />
              )),
            )}
          </Stack>
        ) : null}

        {layerPlan ? <LayerPlanPanel layerPlan={layerPlan} /> : null}

        {componentId ? (
          <HorizontalMaterialsStrip
            componentId={componentId}
            slots={Array.isArray(slots) ? slots : []}
            onStateChange={onStateChange}
            onSlotUpload={onSlotUpload}
            onSlotGenerate={onSlotGenerate}
            onSlotRemoveBackground={onSlotRemoveBackground}
            disabled={disabled}
          />
        ) : (
          <Alert severity="warning">当前布局没有素材区，请回到上一步重新生成编辑界面。</Alert>
        )}
      </CardContent>
    </Card>
  );
};

type HorizontalMaterialsStripProps = {
  componentId: string;
  slots: MediaSlot[];
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  disabled: boolean;
};

const HorizontalMaterialsStrip: React.FC<HorizontalMaterialsStripProps> = ({
  componentId,
  slots,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
        <Typography variant="subtitle2">图层素材</Typography>
        <Typography variant="caption" color="text.secondary">
          可横向滑动浏览所有图层
        </Typography>
      </Stack>
      <Box sx={{ overflowX: 'auto', overflowY: 'visible', pb: 1 }}>
        <Box sx={{ display: 'flex', gap: 2, minWidth: 'max-content' }}>
          {slots.map((slot) => (
            <MaterialSlotInlineCard
              key={slot.id}
              componentId={componentId}
              slots={slots}
              slot={slot}
              onStateChange={onStateChange}
              onSlotUpload={onSlotUpload}
              onSlotGenerate={onSlotGenerate}
              onSlotRemoveBackground={onSlotRemoveBackground}
              disabled={disabled}
            />
          ))}
        </Box>
      </Box>
    </Box>
  );
};

type MaterialSlotInlineCardProps = {
  componentId: string;
  slots: MediaSlot[];
  slot: MediaSlot;
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  disabled: boolean;
};

const MaterialSlotInlineCard: React.FC<MaterialSlotInlineCardProps> = ({
  componentId,
  slots,
  slot,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const expectsTransparent = slot.layerType !== 'background';

  const updateSlot = (updater: (slotValue: MediaSlot) => MediaSlot) => {
    onStateChange(
      componentId,
      slots.map((current) => (current.id === slot.id ? updater(current) : current)),
    );
  };

  return (
    <Card
      variant="outlined"
      sx={{
        width: 320,
        flex: '0 0 320px',
        borderRadius: 2,
        borderColor: 'rgba(15, 23, 42, 0.12)',
        backgroundColor: '#fff',
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
        <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="center">
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="subtitle1" noWrap>
              {slot.label}
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
              <Typography variant="caption" color="text.secondary">
                {slot.layerType}
              </Typography>
              {expectsTransparent ? (
                <Chip
                  size="small"
                  label={slot.hasTransparentBg ? '透明底' : '建议透明底'}
                  color={slot.hasTransparentBg ? 'success' : 'default'}
                />
              ) : null}
            </Stack>
          </Box>
          <Stack direction="row" spacing={0.5}>
            <IconButton onClick={() => fileInputRef.current?.click()} disabled={disabled} title="上传" size="small">
              <UploadFileIcon fontSize="small" />
            </IconButton>
            <IconButton onClick={() => onSlotGenerate(componentId, slot)} disabled={disabled} title="生成" size="small">
              <AutoAwesomeIcon fontSize="small" />
            </IconButton>
            <IconButton
              onClick={() => onSlotRemoveBackground(componentId, slot)}
              disabled={disabled || slot.layerType === 'background' || !slot.uri}
              title="抠图"
              size="small"
            >
              <ContentCutIcon fontSize="small" />
            </IconButton>
            <IconButton
              onClick={() => updateSlot((prev) => ({ ...prev, uri: undefined, hasTransparentBg: false }))}
              disabled={disabled || !slot.uri}
              title="清除"
              size="small"
            >
              <DeleteOutlineIcon fontSize="small" />
            </IconButton>
          </Stack>
        </Stack>

        <TextField
          label="该图层提示词"
          value={slot.prompt || ''}
          onChange={(event) => updateSlot((prev) => ({ ...prev, prompt: event.target.value }))}
          fullWidth
          multiline
          minRows={3}
          disabled={disabled}
          helperText={expectsTransparent ? '主体/装饰层优先按透明底生成。' : '背景层建议写完整场景、景深和光影。'}
        />

        <Box
          sx={{
            minHeight: 164,
            borderRadius: 2,
            border: '1px dashed rgba(15, 23, 42, 0.18)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
            background: slot.uri ? '#f8fafc' : 'linear-gradient(180deg, rgba(248,250,252,1) 0%, rgba(241,245,249,1) 100%)',
          }}
        >
          {slot.uri ? (
            <img
              src={slot.uri}
              alt={slot.label}
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
            />
          ) : (
            <Typography variant="body2" color="text.secondary">
              未准备素材
            </Typography>
          )}
        </Box>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            onSlotUpload(componentId, slot.id, file);
            event.target.value = '';
          }}
        />
      </CardContent>
    </Card>
  );
};

const PreviewCanvas: React.FC<{
  width: number;
  height: number;
  backgroundUri: string | null;
  layers: any[];
}> = ({ width, height, backgroundUri, layers }) => {
  const safeWidth = Math.max(1, width || 1);
  const safeHeight = Math.max(1, height || 1);

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
      <Box
        sx={{
          width: '100%',
          maxHeight: '100%',
          aspectRatio: `${safeWidth} / ${safeHeight}`,
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: '0 24px 60px rgba(15, 23, 42, 0.18)',
          backgroundColor: '#f8fafc',
        }}
      >
        <svg
          viewBox={`0 0 ${safeWidth} ${safeHeight}`}
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid meet"
        >
          <rect x="0" y="0" width={safeWidth} height={safeHeight} fill="#f3f6fa" />
          {backgroundUri ? (
            <image href={backgroundUri} x="0" y="0" width={safeWidth} height={safeHeight} preserveAspectRatio="xMidYMid slice" />
          ) : null}
          {layers.map((layer: any) => {
            const x = Number(layer?.placement?.x ?? 0);
            const y = Number(layer?.placement?.y ?? 0);
            const layerWidth = Number(layer?.width ?? 0);
            const layerHeight = Number(layer?.height ?? 0);
            if ((layer.kind || 'image') === 'image') {
              const href = layer.image_base64?.startsWith('data:')
                ? layer.image_base64
                : `data:image/png;base64,${layer.image_base64 || ''}`;
              return (
                <image
                  key={layer.id || `${layer.name}-${x}-${y}`}
                  href={href}
                  x={x}
                  y={y}
                  width={layerWidth}
                  height={layerHeight}
                  preserveAspectRatio="xMidYMid meet"
                />
              );
            }
            if (layer.kind === 'shape') {
              return (
                <rect
                  key={layer.id || `${layer.name}-${x}-${y}`}
                  x={x}
                  y={y}
                  width={layerWidth}
                  height={layerHeight}
                  rx={Number(layer.style?.radius ?? 24)}
                  ry={Number(layer.style?.radius ?? 24)}
                  fill={layer.style?.fill || 'rgba(255,255,255,0.72)'}
                />
              );
            }
            return (
              <foreignObject
                key={layer.id || `${layer.name}-${x}-${y}`}
                x={x}
                y={y}
                width={layerWidth}
                height={layerHeight}
              >
                <div
                  xmlns="http://www.w3.org/1999/xhtml"
                  style={{
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent:
                      layer.style?.align === 'center'
                        ? 'center'
                        : layer.style?.align === 'right'
                          ? 'flex-end'
                          : 'flex-start',
                    color: layer.style?.color || '#111827',
                    fontSize: `${layer.style?.fontSize || 48}px`,
                    fontWeight: layer.style?.fontWeight || 700,
                    lineHeight: String(layer.style?.lineHeight || 1.2),
                    background: layer.style?.backgroundColor || 'transparent',
                    padding: `${layer.style?.padding || 18}px`,
                    boxSizing: 'border-box',
                    whiteSpace: 'pre-wrap',
                    overflow: 'hidden',
                    textAlign: layer.style?.align || 'left',
                  }}
                >
                  {layer.text || ''}
                </div>
              </foreignObject>
            );
          })}
        </svg>
      </Box>
    </Box>
  );
};

const LayerPlanPanel: React.FC<{ layerPlan: any }> = ({ layerPlan }) => {
  const layers = Array.isArray(layerPlan?.layer_plan) ? layerPlan.layer_plan : [];
  const estimatedLayers = Number(layerPlan?.estimated_layers || layers.length || 0);
  const estimatedTime = Number(layerPlan?.estimated_time_seconds || 0);
  const textRequirement = layerPlan?.meta?.text_requirement || {};

  return (
    <Card variant="outlined">
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <Typography variant="subtitle2">图层规划</Typography>
          <Chip label={`${estimatedLayers} 层`} size="small" />
          {estimatedTime > 0 ? <Chip label={`约 ${estimatedTime}s`} size="small" /> : null}
          {textRequirement?.needs_copy ? <Chip label="含文案区" size="small" color="info" /> : null}
          {textRequirement?.needs_decor ? <Chip label="含装饰层" size="small" /> : null}
        </Stack>
        <Stack spacing={1.25}>
          {layers.map((layer: any) => {
            const placement = formatLayerPlacement(layer?.placement);
            const prompt = typeof layer?.prompt === 'string' ? layer.prompt.trim() : '';
            const needsTransparent = Boolean(layer?.generation_params?.needs_transparent_bg);
            return (
              <Box
                key={layer?.layer_id || layer?.layer_name || prompt}
                sx={{
                  p: 1.25,
                  borderRadius: 1.5,
                  border: '1px solid rgba(15, 23, 42, 0.10)',
                  backgroundColor: 'rgba(248, 250, 252, 0.9)',
                }}
              >
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" flexWrap="wrap" useFlexGap>
                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {typeof layer?.order === 'number' ? `${layer.order + 1}. ` : ''}
                      {layer?.layer_name || layer?.layer_id || '图层'}
                    </Typography>
                    <Chip label={formatLayerType(layer?.layer_type)} size="small" />
                    {needsTransparent ? <Chip label="建议透明底" size="small" color="success" /> : null}
                  </Stack>
                  {placement ? (
                    <Typography variant="caption" color="text.secondary">
                      {placement}
                    </Typography>
                  ) : null}
                </Stack>
                {prompt ? (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
                    {prompt}
                  </Typography>
                ) : null}
              </Box>
            );
          })}
        </Stack>
      </CardContent>
    </Card>
  );
};

function isCopySection(section: LayoutSection): boolean {
  if ((section.cardType || '').toLowerCase() === 'copywriting') return true;
  return (section.components ?? []).some((component: any) => {
    const sample = `${component.id || ''} ${component.label || ''} ${component.title || ''}`.toLowerCase();
    return /标题|headline|subtitle|文案|copy|body/.test(sample);
  });
}

function formatLayerType(layerType: unknown): string {
  const value = typeof layerType === 'string' ? layerType.toLowerCase() : '';
  if (value === 'background') return '背景层';
  if (value === 'subject') return '主体层';
  if (value === 'decor') return '装饰层';
  return '图层';
}

function formatLayerPlacement(placement: any): string {
  if (!placement || typeof placement !== 'object') return '';
  if (typeof placement.w === 'number' && typeof placement.h === 'number') {
    return `安全区 x:${Math.round((placement.x || 0) * 100)}% y:${Math.round((placement.y || 0) * 100)}%`;
  }
  if (typeof placement.x === 'number' && typeof placement.y === 'number') {
    return `位置 x:${Math.round(placement.x * 100)}% y:${Math.round(placement.y * 100)}%`;
  }
  return '';
}

export default SceneEditWorkspace;
