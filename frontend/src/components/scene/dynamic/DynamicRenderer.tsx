import React, { useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  FormControl,
  FormControlLabel,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Stack,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  IconButton,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { api } from '../../../api';

export type LayoutConfig = {
  meta?: Record<string, any>;
  sections?: LayoutSection[];
};

export type LayoutSection = {
  id: string;
  title?: string;
  description?: string;
  cardType?: string;
  layout?: {
    span?: number;
    tone?: 'accent' | 'soft';
  };
  components?: LayoutComponent[];
};

type LayoutComponent =
  | TextInputComponent
  | TextareaComponent
  | SelectComponent
  | MultiSelectComponent
  | NumberInputComponent
  | SliderComponent
  | ToggleComponent
  | RatioSelectComponent
  | ColorPaletteComponent
  | PromptEditorComponent
  | MediaUploaderComponent;

type BaseComponent = {
  id: string;
  label?: string;
  title?: string;
  type: string;
  default?: any;
  helperText?: string;
};

type TextInputComponent = BaseComponent & {
  type: 'text-input';
  placeholder?: string;
};

type TextareaComponent = BaseComponent & {
  type: 'textarea';
  placeholder?: string;
};

type SelectComponent = BaseComponent & {
  type: 'select';
  options: OptionItem[];
  display?: 'chips' | 'cards' | 'images' | 'swatches';
};

type MultiSelectComponent = BaseComponent & {
  type: 'multi-select';
  options: OptionItem[];
  display?: 'chips' | 'cards' | 'images' | 'swatches';
};

type NumberInputComponent = BaseComponent & {
  type: 'number-input';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  helperText?: string;
};

type SliderComponent = BaseComponent & {
  type: 'slider';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  helperText?: string;
};

type ToggleComponent = BaseComponent & {
  type: 'toggle';
  helperText?: string;
};

type RatioSelectComponent = BaseComponent & {
  type: 'ratio-select';
  options: OptionItem[];
  display?: 'chips' | 'cards' | 'images' | 'swatches';
};

type PaletteOption = {
  value: string;
  label?: string;
  color?: string;
};

type VisualOption = {
  value: string;
  label?: string;
  description?: string;
  image?: string;
  previewPrompt?: string;
  color?: string;
};

type OptionItem = string | VisualOption;

type NormalizedOption = {
  value: string;
  label: string;
  description?: string;
  image?: string;
  previewPrompt?: string;
  color?: string;
};

type ColorPaletteComponent = BaseComponent & {
  type: 'color-palette';
  options: Array<PaletteOption | string>;
  allowMultiple?: boolean;
};

type PromptEditorField = {
  id: string;
  label?: string;
  placeholder?: string;
  default?: string;
  helperText?: string;
};

type PromptEditorComponent = BaseComponent & {
  type: 'prompt-editor';
  fields: PromptEditorField[];
};

export type MediaSlot = {
  id: string;
  label: string;
  layerType: 'background' | 'subject' | 'decor';
  prompt: string;
  uri?: string;
  hasTransparentBg?: boolean;
};

type MediaUploaderComponent = BaseComponent & {
  type: 'media-uploader';
  slots: MediaSlot[];
};

type DynamicRendererProps = {
  config: LayoutConfig;
  state: Record<string, any>;
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  disabled?: boolean;
  renderMode?: 'grid' | 'items';
  containerSx?: Record<string, any>;
};

const DynamicRenderer: React.FC<DynamicRendererProps> = ({
  config,
  state,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
  renderMode = 'grid',
  containerSx,
}) => {
  const sections = config.sections ?? [];
  const layoutMeta = (config.meta ?? {}).layout ?? {};
  const columns = coerceNumber(layoutMeta.columns);
  const minCardWidth = coerceNumber(layoutMeta.minCardWidth) ?? 360;
  const gapValue = coerceNumber(layoutMeta.gap) ?? 3;
  const dense = typeof layoutMeta.dense === 'boolean' ? layoutMeta.dense : true;
  const gridTemplateColumns =
    columns && columns > 0
      ? { xs: '1fr', lg: `repeat(${Math.max(1, Math.floor(columns))}, minmax(0, 1fr))` }
      : { xs: '1fr', lg: `repeat(auto-fit, minmax(${Math.max(240, Math.floor(minCardWidth))}px, 1fr))` };

  const renderSectionCard = (section: LayoutSection) => {
    const span = coerceNumber(section.layout?.span);
    const columnSpan = span && span > 0 ? Math.floor(span) : undefined;
    const gridColumn = columnSpan ? { xs: 'auto', lg: `span ${columnSpan}` } : 'auto';
    const tone = section.layout?.tone;
    const cardToneStyle =
      tone === 'accent'
        ? {
            borderColor: 'primary.light',
            backgroundColor: 'rgba(33, 150, 243, 0.05)',
          }
        : tone === 'soft'
          ? {
              backgroundColor: 'rgba(15, 23, 42, 0.03)',
            }
          : {};
    return (
      <SectionCard
        key={section.id}
        section={section}
        cardType={section.cardType}
        gridColumn={gridColumn}
        cardToneStyle={cardToneStyle}
        state={state}
        onStateChange={onStateChange}
        onSlotUpload={onSlotUpload}
        onSlotGenerate={onSlotGenerate}
        onSlotRemoveBackground={onSlotRemoveBackground}
        disabled={disabled}
      />
    );
  };

  if (renderMode === 'items') {
    return <>{sections.map((section) => renderSectionCard(section))}</>;
  }

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns,
        gap: gapValue,
        gridAutoFlow: dense ? 'dense' : 'row',
        alignItems: 'start',
        ...containerSx,
      }}
    >
      {sections.map((section) => renderSectionCard(section))}
    </Box>
  );
};

export default DynamicRenderer;

type SectionCardProps = {
  section: LayoutSection;
  cardType?: string;
  gridColumn?: any;
  cardToneStyle?: Record<string, any>;
  headerActions?: React.ReactNode;
  state: Record<string, any>;
  onStateChange: (componentId: string, value: any) => void;
  onSlotUpload: (componentId: string, slotId: string, file: File) => void;
  onSlotGenerate: (componentId: string, slot: MediaSlot) => void;
  onSlotRemoveBackground: (componentId: string, slot: MediaSlot) => void;
  disabled?: boolean;
};

export const SectionCard: React.FC<SectionCardProps> = ({
  section,
  cardType,
  gridColumn,
  cardToneStyle,
  headerActions,
  state,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
  return (
    <Card
      variant="outlined"
      data-card-type={cardType}
      sx={{
        gridColumn,
        minWidth: 0,
        boxShadow: '0 12px 28px rgba(15, 23, 42, 0.08)',
        borderRadius: 2,
        alignSelf: 'start',
        ...cardToneStyle,
      }}
    >
      <CardContent sx={{ display: 'flex', flexDirection: 'column' }}>
        {section.title || headerActions ? (
          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" sx={{ mb: section.description ? 0.5 : 1 }}>
            {section.title ? (
              <Typography variant="h6">
                {section.title}
              </Typography>
            ) : <Box />}
            {headerActions ? <Box>{headerActions}</Box> : null}
          </Stack>
        ) : null}
        {section.description ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            {section.description}
          </Typography>
        ) : null}
        <Box>
          <Stack spacing={2}>
            {(section.components ?? []).map((component) => (
              <Box key={component.id}>
                <ComponentRenderer
                  component={component}
                  value={state[component.id]}
                  onChange={(value) => onStateChange(component.id, value)}
                  onSlotUpload={(slotId, file) => onSlotUpload(component.id, slotId, file)}
                  onSlotGenerate={(slot) => onSlotGenerate(component.id, slot)}
                  onSlotRemoveBackground={(slot) => onSlotRemoveBackground(component.id, slot)}
                  disabled={disabled}
                />
              </Box>
            ))}
          </Stack>
        </Box>
      </CardContent>
    </Card>
  );
};

type ComponentRendererProps = {
  component: LayoutComponent;
  value: any;
  onChange: (value: any) => void;
  onSlotUpload: (slotId: string, file: File) => void;
  onSlotGenerate: (slot: MediaSlot) => void;
  onSlotRemoveBackground: (slot: MediaSlot) => void;
  disabled?: boolean;
};

type NormalizedPaletteOption = {
  value: string;
  label: string;
  color?: string;
};

const coerceNumber = (input: unknown): number | null => {
  if (typeof input === 'number' && !Number.isNaN(input)) return input;
  if (typeof input === 'string' && input.trim() !== '') {
    const num = Number(input);
    return Number.isNaN(num) ? null : num;
  }
  return null;
};

const normalizePaletteOptions = (options: Array<PaletteOption | string> | undefined): NormalizedPaletteOption[] => {
  if (!options) return [];
  return options
    .map((opt) => {
      if (typeof opt === 'string') {
        const value = opt.trim();
        if (!value) return null;
        return { value, label: value };
      }
      if (!opt?.value) return null;
      const value = String(opt.value).trim();
      if (!value) return null;
      const label = opt.label ? String(opt.label).trim() : value;
      const color = opt.color ? String(opt.color).trim() : undefined;
      return { value, label, color };
    })
    .filter(Boolean) as NormalizedPaletteOption[];
};

const resolvePaletteSwatch = (option: NormalizedPaletteOption): string | undefined => {
  const candidate = option.color || option.value;
  if (!candidate) return undefined;
  if (candidate.startsWith('#')) return candidate;
  if (candidate.startsWith('rgb') || candidate.startsWith('hsl')) return candidate;
  return undefined;
};

const normalizeOptions = (options: OptionItem[] | undefined): NormalizedOption[] => {
  if (!options) return [];
  return options
    .map((opt) => {
      if (typeof opt === 'string') {
        return { value: opt, label: opt };
      }
      const rawValue = opt.value || opt.label;
      if (!rawValue) return null;
      return {
        value: rawValue,
        label: opt.label || rawValue,
        description: opt.description,
        image: opt.image,
        previewPrompt: opt.previewPrompt,
        color: opt.color,
      };
    })
    .filter(Boolean) as NormalizedOption[];
};

type OptionCardProps = {
  option: NormalizedOption;
  selected: boolean;
  onSelect: () => void;
  previewUrl?: string;
  loading?: boolean;
  onGeneratePreview?: () => void;
  disabled?: boolean;
};

const OptionCard: React.FC<OptionCardProps> = ({
  option,
  selected,
  onSelect,
  previewUrl,
  loading,
  onGeneratePreview,
  disabled,
}) => {
  const imageUrl = previewUrl || option.image;
  return (
    <Box
      role="button"
      tabIndex={0}
      onClick={() => {
        if (!disabled) onSelect();
      }}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      sx={{
        border: selected ? '2px solid' : '1px solid',
        borderColor: selected ? 'primary.main' : 'rgba(15, 23, 42, 0.15)',
        borderRadius: 1.5,
        p: 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.6 : 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 0.75,
        position: 'relative',
        backgroundColor: '#fff',
        minHeight: 120,
      }}
    >
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          borderRadius: 1,
          overflow: 'hidden',
          backgroundColor: option.color || 'rgba(15, 23, 42, 0.04)',
          border: '1px solid rgba(15, 23, 42, 0.08)',
          height: 80,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {imageUrl ? (
          <Box
            component="img"
            src={imageUrl}
            alt={option.label}
            sx={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <Typography variant="caption" color="text.secondary">
            {option.color ? '' : '暂无预览'}
          </Typography>
        )}
        {onGeneratePreview && !imageUrl ? (
          <IconButton
            size="small"
            onClick={(event) => {
              event.stopPropagation();
              onGeneratePreview();
            }}
            sx={{
              position: 'absolute',
              right: 4,
              bottom: 4,
              backgroundColor: 'rgba(255,255,255,0.85)',
            }}
            disabled={disabled || loading}
          >
            {loading ? <CircularProgress size={16} /> : <AutoAwesomeIcon fontSize="inherit" />}
          </IconButton>
        ) : null}
      </Box>
      <Box>
        <Typography variant="body2">{option.label}</Typography>
        {option.description ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            {option.description}
          </Typography>
        ) : null}
      </Box>
      {selected ? (
        <Chip
          label="已选"
          size="small"
          color="primary"
          sx={{ position: 'absolute', top: 6, right: 6 }}
        />
      ) : null}
    </Box>
  );
};

const ComponentRenderer: React.FC<ComponentRendererProps> = ({
  component,
  value,
  onChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
  const [optionPreviews, setOptionPreviews] = useState<Record<string, string>>({});
  const [optionLoading, setOptionLoading] = useState<Record<string, boolean>>({});

  const handleGenerateOptionPreview = async (key: string, prompt?: string) => {
    if (!prompt || optionLoading[key]) return;
    setOptionLoading((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await api.post('/generate-image', {
        prompt,
        width: 192,
        height: 192,
        num_inference_steps: 12,
      });
      const base64 = res.data?.image_base64;
      if (base64) {
        const dataUrl = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
        setOptionPreviews((prev) => ({ ...prev, [key]: dataUrl }));
      }
    } catch (error) {
      // Swallow preview errors; user can retry.
    } finally {
      setOptionLoading((prev) => ({ ...prev, [key]: false }));
    }
  };

  switch (component.type) {
    case 'text-input':
      return (
        <TextField
          label={component.label}
          value={value ?? component.default ?? ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={component.placeholder}
          fullWidth
          disabled={disabled}
        />
      );
    case 'textarea':
      return (
        <TextField
          label={component.label}
          value={value ?? component.default ?? ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={component.placeholder}
          fullWidth
          multiline
          minRows={3}
          disabled={disabled}
        />
      );
    case 'select': {
      const options = normalizeOptions(component.options);
      const defaultValue = component.default ?? options[0]?.value ?? '';
      const currentValue = typeof value === 'string' && value ? value : defaultValue;
      const display = component.display;
      if (display === 'chips') {
        return (
          <Box>
            {component.label ? (
              <Typography variant="body2" gutterBottom>
                {component.label}
              </Typography>
            ) : null}
            <ToggleButtonGroup
              exclusive
              value={currentValue}
              onChange={(_, nextValue) => {
                if (nextValue !== null) onChange(nextValue);
              }}
              size="small"
              disabled={disabled}
            >
              {options.map((opt) => (
                <ToggleButton key={opt.value} value={opt.value}>
                  {opt.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
          </Box>
        );
      }
      if (display === 'cards' || display === 'images' || display === 'swatches') {
        return (
          <Box>
            {component.label ? (
              <Typography variant="body2" gutterBottom>
                {component.label}
              </Typography>
            ) : null}
            <Box
              sx={{
                display: 'grid',
                gap: 1,
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              }}
            >
              {options.map((opt) => {
                const key = `${component.id}:${opt.value}`;
                return (
                  <OptionCard
                    key={opt.value}
                    option={opt}
                    selected={currentValue === opt.value}
                    previewUrl={optionPreviews[key]}
                    loading={optionLoading[key]}
                    onGeneratePreview={
                      opt.previewPrompt
                        ? () => handleGenerateOptionPreview(key, opt.previewPrompt)
                        : undefined
                    }
                    onSelect={() => onChange(opt.value)}
                    disabled={disabled}
                  />
                );
              })}
            </Box>
            {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
          </Box>
        );
      }
      return (
        <FormControl fullWidth disabled={disabled}>
          <InputLabel>{component.label}</InputLabel>
          <Select
            label={component.label}
            value={currentValue}
            onChange={(e) => onChange(e.target.value)}
          >
            {options.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
    }
    case 'multi-select': {
      const options = normalizeOptions(component.options);
      const values = Array.isArray(value) ? value : component.default ?? [];
      const display = component.display;
      if (display === 'chips') {
        return (
          <Box>
            {component.label ? (
              <Typography variant="body2" gutterBottom>
                {component.label}
              </Typography>
            ) : null}
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {options.map((opt) => {
                const selected = values.includes(opt.value);
                return (
                  <Chip
                    key={opt.value}
                    label={opt.label}
                    size="small"
                    clickable
                    color={selected ? 'primary' : 'default'}
                    variant={selected ? 'filled' : 'outlined'}
                    onClick={() => {
                      if (disabled) return;
                      const next = selected
                        ? values.filter((v) => v !== opt.value)
                        : [...values, opt.value];
                      onChange(next);
                    }}
                  />
                );
              })}
            </Stack>
            {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
          </Box>
        );
      }
      if (display === 'cards' || display === 'images' || display === 'swatches') {
        return (
          <Box>
            {component.label ? (
              <Typography variant="body2" gutterBottom>
                {component.label}
              </Typography>
            ) : null}
            <Box
              sx={{
                display: 'grid',
                gap: 1,
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              }}
            >
              {options.map((opt) => {
                const key = `${component.id}:${opt.value}`;
                const selected = values.includes(opt.value);
                return (
                  <OptionCard
                    key={opt.value}
                    option={opt}
                    selected={selected}
                    previewUrl={optionPreviews[key]}
                    loading={optionLoading[key]}
                    onGeneratePreview={
                      opt.previewPrompt
                        ? () => handleGenerateOptionPreview(key, opt.previewPrompt)
                        : undefined
                    }
                    onSelect={() => {
                      if (disabled) return;
                      const next = selected
                        ? values.filter((v) => v !== opt.value)
                        : [...values, opt.value];
                      onChange(next);
                    }}
                    disabled={disabled}
                  />
                );
              })}
            </Box>
            {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
          </Box>
        );
      }
      return (
        <FormControl fullWidth disabled={disabled}>
          <InputLabel>{component.label}</InputLabel>
          <Select
            multiple
            label={component.label}
            value={values}
            onChange={(e) => onChange(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
            renderValue={(selected) => (
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {(selected as string[]).map((val) => (
                  <Chip key={val} label={val} size="small" />
                ))}
              </Stack>
            )}
          >
            {options.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
    }
    case 'number-input': {
      const rawValue = coerceNumber(value);
      const fallbackValue = coerceNumber(component.default);
      const currentValue = rawValue ?? fallbackValue ?? '';
      const min = coerceNumber(component.min) ?? undefined;
      const max = coerceNumber(component.max) ?? undefined;
      const step = coerceNumber(component.step) ?? undefined;
      const helperText = component.helperText || (component.unit ? `单位：${component.unit}` : undefined);
      return (
        <TextField
          label={component.label}
          type="number"
          value={currentValue}
          onChange={(e) => {
            const raw = e.target.value;
            onChange(raw === '' ? '' : Number(raw));
          }}
          inputProps={{ min, max, step }}
          helperText={helperText}
          fullWidth
          disabled={disabled}
        />
      );
    }
    case 'slider': {
      const min = coerceNumber(component.min) ?? 0;
      const max = coerceNumber(component.max) ?? 1;
      const defaultValue = coerceNumber(component.default) ?? min;
      const sliderValue = coerceNumber(value) ?? defaultValue;
      const step = coerceNumber(component.step) ?? 1;
      const helperText = component.helperText || (component.unit ? `单位：${component.unit}` : undefined);
      return (
        <Box>
          {component.label ? (
            <Typography variant="body2" gutterBottom>
              {component.label}
            </Typography>
          ) : null}
          <Slider
            value={sliderValue}
            min={min}
            max={max}
            step={step}
            valueLabelDisplay="auto"
            onChange={(_, nextValue) => onChange(nextValue as number)}
            disabled={disabled}
          />
          {helperText ? <FormHelperText>{helperText}</FormHelperText> : null}
        </Box>
      );
    }
    case 'toggle': {
      const checked = typeof value === 'boolean' ? value : Boolean(component.default);
      return (
        <Box>
          <FormControlLabel
            control={
              <Switch
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
                disabled={disabled}
              />
            }
            label={component.label || component.title || '开关'}
          />
          {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
        </Box>
      );
    }
    case 'ratio-select': {
      const options = normalizeOptions(component.options);
      const currentValue =
        typeof value === 'string' && value ? value : component.default ?? options[0]?.value ?? '';
      const display = component.display;
      if (display === 'cards' || display === 'images' || display === 'swatches') {
        return (
          <Box>
            {component.label ? (
              <Typography variant="body2" gutterBottom>
                {component.label}
              </Typography>
            ) : null}
            <Box
              sx={{
                display: 'grid',
                gap: 1,
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
              }}
            >
              {options.map((opt) => {
                const key = `${component.id}:${opt.value}`;
                return (
                  <OptionCard
                    key={opt.value}
                    option={opt}
                    selected={currentValue === opt.value}
                    previewUrl={optionPreviews[key]}
                    loading={optionLoading[key]}
                    onGeneratePreview={
                      opt.previewPrompt
                        ? () => handleGenerateOptionPreview(key, opt.previewPrompt)
                        : undefined
                    }
                    onSelect={() => onChange(opt.value)}
                    disabled={disabled}
                  />
                );
              })}
            </Box>
          </Box>
        );
      }
      return (
        <Box>
          {component.label ? (
            <Typography variant="body2" gutterBottom>
              {component.label}
            </Typography>
          ) : null}
          <ToggleButtonGroup
            exclusive
            value={currentValue}
            onChange={(_, nextValue) => {
              if (nextValue !== null) onChange(nextValue);
            }}
            size="small"
            disabled={disabled}
          >
            {options.map((opt) => (
              <ToggleButton key={opt.value} value={opt.value}>
                {opt.label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
      );
    }
    case 'color-palette': {
      const options = normalizePaletteOptions(component.options);
      const allowMultiple = Boolean(component.allowMultiple);
      const defaultValue = component.default ?? (allowMultiple ? [] : options[0]?.value ?? '');
      const paletteValue = allowMultiple
        ? Array.isArray(value)
          ? value
          : Array.isArray(defaultValue)
            ? defaultValue
            : defaultValue
            ? [defaultValue]
            : []
        : typeof value === 'string'
          ? value
          : typeof defaultValue === 'string'
            ? defaultValue
            : options[0]?.value ?? '';
      return (
        <Box>
          {component.label ? (
            <Typography variant="body2" gutterBottom>
              {component.label}
            </Typography>
          ) : null}
          <ToggleButtonGroup
            value={paletteValue}
            exclusive={!allowMultiple}
            onChange={(_, nextValue) => {
              if (nextValue !== null) onChange(nextValue);
            }}
            size="small"
            disabled={disabled}
          >
            {options.map((opt) => {
              const swatch = resolvePaletteSwatch(opt);
              return (
                <ToggleButton key={opt.value} value={opt.value} sx={{ gap: 1, px: 1.25 }}>
                  {swatch ? (
                    <Box
                      sx={{
                        width: 14,
                        height: 14,
                        borderRadius: '50%',
                        backgroundColor: swatch,
                        border: '1px solid rgba(0,0,0,0.15)',
                      }}
                    />
                  ) : null}
                  <Typography variant="caption">{opt.label}</Typography>
                </ToggleButton>
              );
            })}
          </ToggleButtonGroup>
          {component.helperText ? <FormHelperText>{component.helperText}</FormHelperText> : null}
        </Box>
      );
    }
    case 'prompt-editor': {
      const fields = component.fields ?? [];
      const currentValue = value && typeof value === 'object' ? value : {};
      const title = component.title || component.label;
      return (
        <Stack spacing={1.5}>
          {title ? (
            <Typography variant="subtitle2" gutterBottom>
              {title}
            </Typography>
          ) : null}
          {fields.map((field) => (
            <TextField
              key={field.id}
              label={field.label}
              placeholder={field.placeholder}
              value={currentValue[field.id] ?? field.default ?? ''}
              onChange={(e) => onChange({ ...currentValue, [field.id]: e.target.value })}
              fullWidth
              multiline
              minRows={2}
              disabled={disabled}
              helperText={field.helperText}
            />
          ))}
        </Stack>
      );
    }
    case 'media-uploader':
      return (
        <MediaUploaderField
          title={component.title || component.label || '素材'}
          slots={Array.isArray(value) ? value : Array.isArray(component.slots) ? component.slots : []}
          onSlotChange={onChange}
          onSlotUpload={onSlotUpload}
          onSlotGenerate={onSlotGenerate}
          onSlotRemoveBackground={onSlotRemoveBackground}
          disabled={disabled}
        />
      );
    default:
      return null;
  }
};

type MediaUploaderFieldProps = {
  title: string;
  slots: MediaSlot[];
  onSlotChange: (slots: MediaSlot[]) => void;
  onSlotUpload: (slotId: string, file: File) => void;
  onSlotGenerate: (slot: MediaSlot) => void;
  onSlotRemoveBackground: (slot: MediaSlot) => void;
  disabled?: boolean;
};

const MediaUploaderField: React.FC<MediaUploaderFieldProps> = ({
  title,
  slots,
  onSlotChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadTarget, setUploadTarget] = useState<string | null>(null);
  const safeSlots = Array.isArray(slots) ? slots : [];

  const handlePickUpload = (slotId: string) => {
    setUploadTarget(slotId);
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length || !uploadTarget) return;
    onSlotUpload(uploadTarget, files[0]);
    event.target.value = '';
    setUploadTarget(null);
  };

  const updateSlot = (slotId: string, updater: (slot: MediaSlot) => MediaSlot) => {
    const next = safeSlots.map((s) => (s.id === slotId ? updater(s) : s));
    onSlotChange(next);
  };

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {title}
      </Typography>
      <Stack spacing={2}>
        {!safeSlots.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                当前没有可用的素材图层。通常是布局返回了 `media-uploader`，但没有带上 `slots`。
              </Typography>
            </CardContent>
          </Card>
        ) : null}
        {safeSlots.map((slot) => {
          const expectsTransparent = slot.layerType !== 'background';
          return (
            <Card key={slot.id} variant="outlined">
              <CardContent>
                <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography variant="subtitle1">{slot.label}</Typography>
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
                  <Stack direction="row" spacing={1}>
                    <IconButton
                      onClick={() => handlePickUpload(slot.id)}
                      disabled={disabled}
                      title="上传"
                    >
                      <UploadFileIcon />
                    </IconButton>
                    <IconButton
                      onClick={() => onSlotGenerate(slot)}
                      disabled={disabled}
                      title="生成"
                    >
                      <AutoAwesomeIcon />
                    </IconButton>
                    <IconButton
                      onClick={() => onSlotRemoveBackground(slot)}
                      disabled={disabled || slot.layerType === 'background' || !slot.uri}
                      title="抠图"
                    >
                      <ContentCutIcon />
                    </IconButton>
                    <IconButton
                      onClick={() => updateSlot(slot.id, (prev) => ({ ...prev, uri: undefined, hasTransparentBg: false }))}
                      disabled={disabled || !slot.uri}
                      title="清除"
                    >
                      <DeleteOutlineIcon />
                    </IconButton>
                  </Stack>
                </Stack>

                <TextField
                  label="该图层提示词"
                  value={slot.prompt || ''}
                  onChange={(event) => updateSlot(slot.id, (prev) => ({ ...prev, prompt: event.target.value }))}
                  fullWidth
                  multiline
                  minRows={2}
                  disabled={disabled}
                  sx={{ mt: 1.5 }}
                  helperText={
                    expectsTransparent
                      ? '主体/装饰层会优先按透明底生成，便于后续拼版。'
                      : '背景层建议写完整场景、景深和光影。'
                  }
                />

                {slot.uri ? (
                  <Box sx={{ mt: 1 }}>
                    <img
                      src={slot.uri}
                      alt={slot.label}
                      style={{ maxWidth: '100%', borderRadius: 8 }}
                    />
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    未准备素材（可上传或生成）
                  </Typography>
                )}
              </CardContent>
            </Card>
          );
        })}
      </Stack>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={handleFileChange}
      />
    </Box>
  );
};
