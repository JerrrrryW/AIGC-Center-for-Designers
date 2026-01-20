import React, { useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
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
  components?: LayoutComponent[];
};

type LayoutComponent =
  | TextInputComponent
  | TextareaComponent
  | SelectComponent
  | MultiSelectComponent
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

  return (
    <Stack spacing={3}>
      {sections.map((section) => (
        <Card key={section.id} variant="outlined">
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
      ))}
    </Stack>
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

