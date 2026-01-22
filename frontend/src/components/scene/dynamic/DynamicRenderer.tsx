import React, { useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
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

type LayoutConfig = {
  meta?: Record<string, any>;
  sections?: LayoutSection[];
};

type LayoutSection = {
  id: string;
  title?: string;
  layout?: {
    span?: number;
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
  options: string[];
};

type MultiSelectComponent = BaseComponent & {
  type: 'multi-select';
  options: string[];
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
  options: string[];
};

type PaletteOption = {
  value: string;
  label?: string;
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
};

const DynamicRenderer: React.FC<DynamicRendererProps> = ({
  config,
  state,
  onStateChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
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

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns,
        gap: gapValue,
        gridAutoFlow: dense ? 'dense' : 'row',
        alignItems: 'start',
      }}
    >
      {sections.map((section) => {
        const span = coerceNumber(section.layout?.span);
        const columnSpan = span && span > 0 ? Math.floor(span) : undefined;
        const gridColumn = columnSpan ? { xs: 'auto', lg: `span ${columnSpan}` } : 'auto';
        return (
          <Card
            key={section.id}
            variant="outlined"
            sx={{
              gridColumn,
              minWidth: 0,
            }}
          >
            <CardContent>
              {section.title ? (
                <Typography variant="h6" gutterBottom>
                  {section.title}
                </Typography>
              ) : null}
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
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
};

export default DynamicRenderer;

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

const ComponentRenderer: React.FC<ComponentRendererProps> = ({
  component,
  value,
  onChange,
  onSlotUpload,
  onSlotGenerate,
  onSlotRemoveBackground,
  disabled,
}) => {
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
    case 'select':
      return (
        <FormControl fullWidth disabled={disabled}>
          <InputLabel>{component.label}</InputLabel>
          <Select
            label={component.label}
            value={value ?? component.default ?? component.options?.[0] ?? ''}
            onChange={(e) => onChange(e.target.value)}
          >
            {(component.options ?? []).map((opt) => (
              <MenuItem key={opt} value={opt}>
                {opt}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
    case 'multi-select': {
      const values = Array.isArray(value) ? value : component.default ?? [];
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
            {(component.options ?? []).map((opt) => (
              <MenuItem key={opt} value={opt}>
                {opt}
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
      const options = component.options ?? [];
      const currentValue =
        typeof value === 'string' && value ? value : component.default ?? options[0] ?? '';
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
              <ToggleButton key={opt} value={opt}>
                {opt}
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
          slots={Array.isArray(value) ? value : component.slots}
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
    const next = slots.map((s) => (s.id === slotId ? updater(s) : s));
    onSlotChange(next);
  };

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {title}
      </Typography>
      <Stack spacing={2}>
        {slots.map((slot) => (
          <Card key={slot.id} variant="outlined">
            <CardContent>
              <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography variant="subtitle1">{slot.label}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {slot.layerType}
                  </Typography>
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
                </Stack>
              </Stack>
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
        ))}
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
