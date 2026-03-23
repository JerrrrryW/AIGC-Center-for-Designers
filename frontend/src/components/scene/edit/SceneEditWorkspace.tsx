import React, { useMemo } from 'react';
import {
  Alert,
  Backdrop,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Collapse,
  CircularProgress,
  Container,
  Divider,
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
import TuneIcon from '@mui/icons-material/Tune';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { ComponentRenderer, type LayoutSection, type MediaSlot } from '../dynamic/DynamicRenderer';
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
  previewBackgroundUri: string | null;
  previewItemsCount: number;
  previewCanvasWidth: number;
  previewCanvasHeight: number;
  hasPendingGlobalChanges: boolean;
  snapshotWarnings: string[];
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
  onRegenerateScene: () => void;
  onLayerPlacementChange: (layerId: string, placement: { x: number; y: number; w: number; h: number; zIndex: number }) => void;
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
  previewBackgroundUri,
  previewItemsCount,
  previewCanvasWidth,
  previewCanvasHeight,
  hasPendingGlobalChanges,
  snapshotWarnings,
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
  onRegenerateScene,
  onLayerPlacementChange,
  onEnterCanvas,
  enterCanvasDisabled,
}) => {
  const filledSlotCount = mediaSlots.filter((slot) => slot.uri).length;
  const copySections = useMemo(
    () => configSections.filter((section) => isCopySection(section)),
    [configSections],
  );
  const settingSections = useMemo(
    () => configSections.filter((section) => !isCopySection(section)),
    [configSections],
  );
  const coreSettingSections = useMemo(
    () => settingSections.filter((section) => !isAdvancedSettingSection(section)),
    [settingSections],
  );
  const advancedSettingSections = useMemo(
    () => settingSections.filter((section) => isAdvancedSettingSection(section)),
    [settingSections],
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
            <Typography variant="h6">编辑</Typography>
            <Chip label={previewSyncLabel} size="small" color={isPreviewSyncing ? 'default' : 'success'} />
          </Stack>

          {status.type === 'error' && status.message ? (
            <Alert severity="error">
              {status.message}
            </Alert>
          ) : null}

          <SceneSetupCard
            materialsSection={materialsSection}
            uiState={uiState}
            sceneDraft={sceneDraft}
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
                description="文案集中编辑，减少来回查找。"
              />
              <SectionGroupCard
                title="文案编辑"
                subtitle="标题、副标题、正文和主色调集中在一张卡里。"
                sections={copySections}
                state={uiState}
                onStateChange={onStateChange}
                onSlotUpload={onSlotUpload}
                onSlotGenerate={onSlotGenerate}
                onSlotRemoveBackground={onSlotRemoveBackground}
                disabled={isWorking}
                compact
                defaultExpanded
              />
            </>
          ) : null}

          {coreSettingSections.length || advancedSettingSections.length ? (
            <>
              <SectionLabel
                title="参数配置"
                description="常用项放前，高级项折叠收纳。"
              />
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2,
                }}
              >
                {coreSettingSections.length ? (
                  <SectionGroupCard
                    title="核心参数"
                    subtitle="优先保留画幅、风格、氛围和构图。"
                    sections={coreSettingSections}
                    state={uiState}
                    onStateChange={onStateChange}
                    onSlotUpload={onSlotUpload}
                    onSlotGenerate={onSlotGenerate}
                    onSlotRemoveBackground={onSlotRemoveBackground}
                    disabled={isWorking}
                    compact
                    defaultExpanded
                  />
                ) : null}
                {advancedSettingSections.length ? (
                  <SectionGroupCard
                    title="高级参数"
                    subtitle="提示词和技术参数收进一张卡，默认折叠。"
                    sections={advancedSettingSections}
                    state={uiState}
                    onStateChange={onStateChange}
                    onSlotUpload={onSlotUpload}
                    onSlotGenerate={onSlotGenerate}
                    onSlotRemoveBackground={onSlotRemoveBackground}
                    disabled={isWorking}
                    compact
                    defaultExpanded={false}
                    accent="soft"
                  />
                ) : null}
              </Box>
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
              borderRadius: 3,
            }}
          >
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1, minHeight: 0 }}>
              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                <Typography variant="h6">实时预览</Typography>
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
                  backgroundUri={previewBackgroundUri}
                  layers={previewComposition?.layers || []}
                  onLayerPlacementChange={onLayerPlacementChange}
                />
                {isPreviewSyncing ? (
                  <LinearProgress sx={{ position: 'absolute', left: 0, right: 0, top: 0 }} />
                ) : null}
                {hasPendingGlobalChanges ? (
                  <Box
                    sx={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(15, 23, 42, 0.36)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      zIndex: 3,
                      p: 2,
                    }}
                  >
                    <Card
                      sx={{
                        px: 2.5,
                        py: 2,
                        minWidth: 280,
                        maxWidth: 360,
                        textAlign: 'center',
                        borderRadius: 3,
                        boxShadow: '0 24px 60px rgba(15, 23, 42, 0.24)',
                      }}
                    >
                      <Stack spacing={1.5} alignItems="center">
                        <AutoAwesomeIcon color="primary" />
                        <Typography variant="subtitle1">全局配置已变更</Typography>
                        <Typography variant="body2" color="text.secondary">
                          当前预览还是旧版本。点击后会重新生成整张参考图，并自动拆层与补背景。
                        </Typography>
                        <Button variant="contained" onClick={onRegenerateScene} disabled={isWorking}>
                          重新生成整图
                        </Button>
                      </Stack>
                    </Card>
                  </Box>
                ) : null}
              </Box>

              {previewError || snapshotWarnings.length ? (
                <Alert severity={previewError ? 'warning' : 'info'}>
                  {previewError || snapshotWarnings[0]}
                </Alert>
              ) : null}

              {textLayers.length ? (
                <Card variant="outlined">
                  <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Typography variant="subtitle2">导出文字</Typography>
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
                <Alert severity="info">当前没有导出文字。</Alert>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 2 }}>
                <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                  <Chip label="预览自动同步" size="small" variant="outlined" />
                  <Button variant="outlined" onClick={onGenerateAllSlots} disabled={isWorking}>
                    重新识别图层
                  </Button>
                </Stack>
                <Stack direction="row" spacing={1}>
                  <Button variant="outlined" onClick={onRegenerateScene} disabled={isWorking}>
                    重新生成整图
                  </Button>
                  <Button variant="contained" onClick={onEnterCanvas} disabled={enterCanvasDisabled}>
                    进入画布
                  </Button>
                </Stack>
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
      px: 1.25,
      py: 0.9,
      borderRadius: 2,
      borderLeft: '3px solid',
      borderColor: 'rgba(25, 118, 210, 0.36)',
      background: 'rgba(248, 250, 252, 0.88)',
    }}
  >
    <Typography variant="subtitle1">{title}</Typography>
    <Typography variant="caption" color="text.secondary">
      {description}
    </Typography>
  </Box>
);

type SectionGroupCardProps = {
  title: string;
  subtitle?: string;
  sections: LayoutSection[];
  state: Record<string, any>;
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  disabled?: boolean;
  compact?: boolean;
  defaultExpanded?: boolean;
  accent?: 'default' | 'soft';
};

const SectionGroupCard: React.FC<SectionGroupCardProps> = ({
  title,
  subtitle,
  sections,
  state,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
  compact = false,
  defaultExpanded = true,
  accent = 'default',
}) => {
  const [expanded, setExpanded] = React.useState(defaultExpanded);
  const sectionCount = sections.length;
  const configuredCount = sections.reduce((count, section) => count + countDirtyComponents(section, state), 0);

  return (
    <Card
      variant="outlined"
      sx={{
        borderRadius: 2.5,
        boxShadow: accent === 'soft' ? '0 10px 26px rgba(15, 23, 42, 0.07)' : '0 14px 30px rgba(15, 23, 42, 0.09)',
        borderColor: 'rgba(15, 23, 42, 0.10)',
        background: accent === 'soft'
          ? 'linear-gradient(180deg, rgba(250,251,253,1) 0%, rgba(255,255,255,1) 100%)'
          : 'linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%)',
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <Stack direction="row" spacing={1.5} alignItems="flex-start" justifyContent="space-between">
          <Box sx={{ minWidth: 0 }}>
            <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
              <Typography variant="h6">{title}</Typography>
              <Chip label={`${sectionCount} 组`} size="small" variant="outlined" />
              {configuredCount > 0 ? <Chip label={`已配置 ${configuredCount}`} size="small" color="primary" /> : null}
            </Stack>
            {subtitle ? (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            ) : null}
          </Box>
          {!defaultExpanded ? (
            <Button
              size="small"
              startIcon={<TuneIcon fontSize="small" />}
              endIcon={expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
              onClick={() => setExpanded((prev) => !prev)}
            >
              {expanded ? '收起' : '展开'}
            </Button>
          ) : null}
        </Stack>

        <Collapse in={expanded} timeout="auto" unmountOnExit={false}>
          <Stack spacing={compact ? 1.5 : 2}>
            {sections.map((section, index) => (
              <Box key={section.id}>
                {index > 0 ? <Divider sx={{ mb: compact ? 1.5 : 2 }} /> : null}
                <Stack spacing={compact ? 1.25 : 1.75}>
                  {section.title ? (
                    <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
                      <Typography variant="subtitle2">{section.title}</Typography>
                      {section.description ? (
                        <Typography variant="caption" color="text.secondary">
                          {section.description}
                        </Typography>
                      ) : null}
                    </Stack>
                  ) : section.description ? (
                    <Typography variant="caption" color="text.secondary">
                      {section.description}
                    </Typography>
                  ) : null}
                  <Box sx={{ overflowX: 'auto', overflowY: 'visible', pb: 0.5 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        gap: compact ? 1.5 : 2,
                        minWidth: 0,
                        width: '100%',
                        alignItems: 'stretch',
                        flexWrap: 'wrap',
                      }}
                    >
                      {(section.components ?? []).map((component) => (
                        <Box
                          key={component.id}
                          sx={{
                            minWidth: resolveComponentMinWidth(component),
                            width: resolveComponentStretch(component),
                            maxWidth: resolveComponentStretch(component) === '100%' ? 960 : 'none',
                            flex: resolveComponentStretch(component) === '100%' ? '1 0 100%' : '1 1 320px',
                          }}
                        >
                          <ComponentRenderer
                            component={component as any}
                            value={state[component.id]}
                            onChange={(value) => onStateChange(component.id, value)}
                            onSlotUpload={(slotId, file) => onSlotUpload(component.id, slotId, file)}
                            onSlotGenerate={(slot) => onSlotGenerate(component.id, slot)}
                            onSlotRemoveBackground={(slot) => onSlotRemoveBackground(component.id, slot)}
                            disabled={disabled}
                          />
                        </Box>
                      ))}
                    </Box>
                  </Box>
                </Stack>
              </Box>
            ))}
          </Stack>
        </Collapse>
      </CardContent>
    </Card>
  );
};

type SceneSetupCardProps = {
  materialsSection?: LayoutSection;
  uiState: Record<string, any>;
  sceneDraft?: SceneDraft;
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
        border: '1px solid rgba(15, 23, 42, 0.08)',
        background: 'linear-gradient(180deg, rgba(250,252,255,0.98) 0%, rgba(255,255,255,1) 100%)',
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.75 }}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} alignItems={{ xs: 'flex-start', md: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="h6">素材与总览</Typography>
            <Typography variant="caption" color="text.secondary">先看整体，再处理图层。</Typography>
          </Box>
          <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
            <Chip label={`素材 ${filledSlotCount}/${mediaSlots.length}`} size="small" />
            <Button variant="outlined" size="small" onClick={onGenerateAllSlots} disabled={disabled}>
              重新识别图层
            </Button>
          </Stack>
        </Stack>

        <Typography variant="body2" color="text.secondary">
          {sceneDraft?.brief?.summary || '编辑信息已就绪，可以开始准备素材和文案。'}
        </Typography>

        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {sceneDraft?.brief?.aspect_ratio ? <Chip label={`画幅 ${sceneDraft.brief.aspect_ratio}`} size="small" /> : null}
          {sceneDraft?.materials?.slots?.length ? <Chip label={`${sceneDraft.materials.slots.length} 个图层`} size="small" /> : null}
          <Chip label={`文案字段 ${copyReadyCount}/3`} size="small" />
          {layerPlan?.estimated_time_seconds ? <Chip label={`约 ${layerPlan.estimated_time_seconds}s`} size="small" variant="outlined" /> : null}
          {sceneDraft?.reference?.image_base64 ? (
            <Chip label={`参考构图 · ${sceneDraft.reference.provider}`} size="small" color="info" variant="outlined" />
          ) : null}
        </Stack>

        {sceneDraft?.reference?.image_base64 ? (
          <Box
            sx={{
              width: 156,
              borderRadius: 2,
              overflow: 'hidden',
              border: '1px solid rgba(15, 23, 42, 0.08)',
              backgroundColor: 'rgba(248,250,252,0.88)',
            }}
          >
            <img
              src={
                sceneDraft.reference.image_base64.startsWith('data:')
                  ? sceneDraft.reference.image_base64
                  : `data:image/png;base64,${sceneDraft.reference.image_base64}`
              }
              alt="参考构图"
              style={{ display: 'block', width: '100%', height: 'auto' }}
            />
          </Box>
        ) : null}

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
        <Chip label="可横向滑动" size="small" variant="outlined" />
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
  const isBackground = slot.layerType === 'background';

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
        width: 304,
        flex: '0 0 304px',
        borderRadius: 2.5,
        borderColor: 'rgba(15, 23, 42, 0.12)',
        background: 'linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%)',
        boxShadow: '0 8px 22px rgba(15, 23, 42, 0.06)',
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
        <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="center">
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="subtitle1" noWrap>
              {slot.label}
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
              <Chip size="small" label={formatLayerType(slot.layerType)} variant="outlined" />
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
            <IconButton
              onClick={() => onSlotGenerate(componentId, slot)}
              disabled={disabled || isBackground}
              title={isBackground ? '背景跟随整图生成' : '替换当前对象'}
              size="small"
            >
              <AutoAwesomeIcon fontSize="small" />
            </IconButton>
            <IconButton
              onClick={() => onSlotRemoveBackground(componentId, slot)}
              disabled={disabled || isBackground || !slot.uri}
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
          label="提示词"
          value={slot.prompt || ''}
          onChange={(event) => updateSlot((prev) => ({ ...prev, prompt: event.target.value }))}
          fullWidth
          multiline
          minRows={3}
          disabled={disabled}
          helperText={isBackground ? '背景由整图统一生成。' : expectsTransparent ? '点击魔法棒会替换当前对象。' : '写完整场景。'}
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
              暂无素材
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
  onLayerPlacementChange: (layerId: string, placement: { x: number; y: number; w: number; h: number; zIndex: number }) => void;
}> = ({ width, height, backgroundUri, layers, onLayerPlacementChange }) => {
  const safeWidth = Math.max(1, width || 1);
  const safeHeight = Math.max(1, height || 1);
  const frameRef = React.useRef<HTMLDivElement | null>(null);
  const dragRef = React.useRef<{
    layerId: string;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    placement: { x: number; y: number; w: number; h: number; zIndex: number };
  } | null>(null);

  React.useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!dragRef.current || !frameRef.current) return;
      const rect = frameRef.current.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dx = (event.clientX - dragRef.current.startX) / rect.width;
      const dy = (event.clientY - dragRef.current.startY) / rect.height;
      const nextX = clamp01(dragRef.current.originX + dx, 1 - dragRef.current.placement.w);
      const nextY = clamp01(dragRef.current.originY + dy, 1 - dragRef.current.placement.h);
      onLayerPlacementChange(dragRef.current.layerId, {
        ...dragRef.current.placement,
        x: nextX,
        y: nextY,
      });
    };
    const handlePointerUp = () => {
      dragRef.current = null;
    };
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [onLayerPlacementChange]);

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
      <Box
        ref={frameRef}
        sx={{
          width: '100%',
          maxHeight: '100%',
          aspectRatio: `${safeWidth} / ${safeHeight}`,
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: '0 24px 60px rgba(15, 23, 42, 0.18)',
          backgroundColor: '#f8fafc',
          position: 'relative',
        }}
      >
        <Box sx={{ position: 'absolute', inset: 0, background: '#f3f6fa' }} />
        {backgroundUri ? (
          <Box
            component="img"
            src={backgroundUri}
            alt="preview-background"
            sx={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              userSelect: 'none',
              WebkitUserDrag: 'none',
            }}
          />
        ) : null}
        {layers.map((layer: any) => {
          const placement = layer?.placement || {};
          const left = `${Number(placement.x ?? 0) * 100}%`;
          const top = `${Number(placement.y ?? 0) * 100}%`;
          const layerWidth = `${Number(placement.w ?? 0) * 100}%`;
          const layerHeight = `${Number(placement.h ?? 0) * 100}%`;

          if ((layer.kind || 'image') === 'image') {
            return (
              <Box
                key={layer.id || `${layer.name}-${left}-${top}`}
                sx={{
                  position: 'absolute',
                  left,
                  top,
                  width: layerWidth,
                  height: layerHeight,
                  zIndex: Number(placement.zIndex ?? 1),
                  cursor: layer.draggable ? 'grab' : 'default',
                  userSelect: 'none',
                }}
                onPointerDown={(event) => {
                  if (!layer.draggable || !layer.slotId) return;
                  event.preventDefault();
                  dragRef.current = {
                    layerId: layer.slotId,
                    startX: event.clientX,
                    startY: event.clientY,
                    originX: Number(placement.x ?? 0),
                    originY: Number(placement.y ?? 0),
                    placement: {
                      x: Number(placement.x ?? 0),
                      y: Number(placement.y ?? 0),
                      w: Number(placement.w ?? 0),
                      h: Number(placement.h ?? 0),
                      zIndex: Number(placement.zIndex ?? 1),
                    },
                  };
                }}
              >
                <Box
                  component="img"
                  src={layer.dataUrl}
                  alt={layer.name || layer.id || 'preview-layer'}
                  sx={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: 'block',
                    pointerEvents: 'none',
                    filter: 'drop-shadow(0 18px 26px rgba(15, 23, 42, 0.18))',
                    userSelect: 'none',
                    WebkitUserDrag: 'none',
                  }}
                />
              </Box>
            );
          }

          if (layer.kind === 'shape') {
            return (
              <Box
                key={layer.id || `${layer.name}-${left}-${top}`}
                sx={{
                  position: 'absolute',
                  left,
                  top,
                  width: layerWidth,
                  height: layerHeight,
                  zIndex: Number(placement.zIndex ?? 1),
                  borderRadius: `${Number(layer.style?.radius ?? 24)}px`,
                  background: layer.style?.fill || 'rgba(255,255,255,0.72)',
                  boxShadow: '0 16px 40px rgba(15, 23, 42, 0.08)',
                }}
              />
            );
          }

          return (
            <Box
              key={layer.id || `${layer.name}-${left}-${top}`}
              sx={{
                position: 'absolute',
                left,
                top,
                width: layerWidth,
                height: layerHeight,
                zIndex: Number(placement.zIndex ?? 1),
                color: layer.style?.color || '#111827',
                fontSize: `${layer.style?.fontSize || 48}px`,
                fontWeight: layer.style?.fontWeight || 700,
                lineHeight: String(layer.style?.lineHeight || 1.2),
                background: layer.style?.backgroundColor || 'transparent',
                p: `${layer.style?.padding || 0}px`,
                boxSizing: 'border-box',
                whiteSpace: 'pre-wrap',
                overflow: 'hidden',
                textAlign: layer.style?.align || 'left',
                display: 'flex',
                justifyContent:
                  layer.style?.align === 'center'
                    ? 'center'
                    : layer.style?.align === 'right'
                      ? 'flex-end'
                      : 'flex-start',
                alignItems: 'flex-start',
              }}
            >
              {layer.text || ''}
            </Box>
          );
        })}
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

function isAdvancedSettingSection(section: LayoutSection): boolean {
  const sample = `${section.id || ''} ${section.title || ''} ${section.cardType || ''}`.toLowerCase();
  if (/prompt|advanced|control|seed|cfg|negative|steps/.test(sample)) return true;
  return (section.components ?? []).some((component: any) => {
    const componentSample = `${component.id || ''} ${component.label || ''} ${component.title || ''} ${component.type || ''}`.toLowerCase();
    return /prompt|seed|cfg|negative|steps/.test(componentSample);
  });
}

function resolveComponentStretch(component: any): string {
  const type = String(component?.type || '').toLowerCase();
  if (type === 'textarea' || type === 'prompt-editor' || type === 'media-uploader') {
    return '100%';
  }
  const sample = `${component?.id || ''} ${component?.label || ''} ${component?.title || ''}`.toLowerCase();
  if (/正文|body|copy|prompt|主色调|palette|画风|风格|氛围|构图/.test(sample)) {
    return '100%';
  }
  return 'auto';
}

function resolveComponentMinWidth(component: any): number {
  const type = String(component?.type || '').toLowerCase();
  if (type === 'slider') return 280;
  if (type === 'toggle') return 220;
  if (type === 'color-palette' || type === 'ratio-select') return 320;
  if (type === 'select' || type === 'multi-select') return 320;
  if (type === 'textarea' || type === 'prompt-editor') return 540;
  return 260;
}

function countDirtyComponents(section: LayoutSection, state: Record<string, any>): number {
  return (section.components ?? []).reduce((count, component: any) => {
    const value = state[component.id];
    if (value === undefined || value === null) return count;
    if (typeof value === 'string') {
      return value.trim() ? count + 1 : count;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return count + 1;
    }
    if (Array.isArray(value)) {
      return value.length ? count + 1 : count;
    }
    if (typeof value === 'object') {
      return Object.values(value).some((item) => {
        if (typeof item === 'string') return item.trim();
        if (Array.isArray(item)) return item.length > 0;
        return Boolean(item);
      })
        ? count + 1
        : count;
    }
    return count;
  }, 0);
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

function clamp01(value: number, max = 1): number {
  return Math.max(0, Math.min(max, value));
}

export default SceneEditWorkspace;
