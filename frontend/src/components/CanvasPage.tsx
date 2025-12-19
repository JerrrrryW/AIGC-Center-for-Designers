import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import FlipToFrontIcon from '@mui/icons-material/FlipToFront';
import FlipToBackIcon from '@mui/icons-material/FlipToBack';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import DownloadIcon from '@mui/icons-material/Download';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloseIcon from '@mui/icons-material/Close';
import InferencePanel from './inference/InferencePanel';

type CanvasItem = {
  id: string;
  src: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  naturalWidth: number;
  naturalHeight: number;
};

const CANVAS_DIMENSION = 4000;
const MIN_ITEM_SIZE = 48;
const MAX_ZOOM = 2;
const MIN_ZOOM = 0.25;
const ZOOM_STEP = 0.1;

const CanvasPage: React.FC = () => {
  const [items, setItems] = useState<CanvasItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [isExporting, setIsExporting] = useState(false);
  const [isInferenceOpen, setIsInferenceOpen] = useState(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragState = useRef<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const resizeState = useRef<{
    id: string;
    originX: number;
    originY: number;
    startWidth: number;
    startHeight: number;
  } | null>(null);
  const createdUrls = useRef<Set<string>>(new Set());
  const zoomRef = useRef(zoom);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    // Center the scrollable area on the first load.
    container.scrollLeft = (CANVAS_DIMENSION - container.clientWidth) / 2;
    container.scrollTop = (CANVAS_DIMENSION - container.clientHeight) / 2;
  }, []);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!canvasRef.current) {
        return;
      }

      const canvasBounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - canvasBounds.left) / zoomRef.current;
      const pointerY = (event.clientY - canvasBounds.top) / zoomRef.current;

      if (dragState.current) {
        const { id, offsetX, offsetY } = dragState.current;
        setItems((prev) =>
          prev.map((item) => {
            if (item.id !== id) {
              return item;
            }
            const newX = clamp(pointerX - offsetX, 0, CANVAS_DIMENSION - item.width);
            const newY = clamp(pointerY - offsetY, 0, CANVAS_DIMENSION - item.height);
            return { ...item, x: newX, y: newY };
          }),
        );
      } else if (resizeState.current) {
        const { id, originX, originY, startWidth, startHeight } = resizeState.current;
        setItems((prev) =>
          prev.map((item) => {
            if (item.id !== id) {
              return item;
            }
            const deltaX = pointerX - originX;
            const deltaY = pointerY - originY;
            const nextWidth = clamp(startWidth + deltaX, MIN_ITEM_SIZE, CANVAS_DIMENSION - item.x);
            const nextHeight = clamp(
              startHeight + deltaY,
              MIN_ITEM_SIZE,
              CANVAS_DIMENSION - item.y,
            );
            return {
              ...item,
              width: nextWidth,
              height: nextHeight,
            };
          }),
        );
      }
    };

    const handleMouseUp = () => {
      dragState.current = null;
      resizeState.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  useEffect(() => {
    return () => {
      createdUrls.current.forEach((url) => {
        URL.revokeObjectURL(url);
      });
    };
  }, []);

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.zIndex - b.zIndex),
    [items],
  );

  const addCanvasItem = useCallback(
    (src: string, name: string, naturalWidth: number, naturalHeight: number) => {
      if (!naturalWidth || !naturalHeight) {
        return;
      }

      const baseWidth = Math.min(naturalWidth, 320);
      const scale = baseWidth / naturalWidth;
      const baseHeight = naturalHeight * scale;
      const canvasCenter = CANVAS_DIMENSION / 2;

      const itemTemplate: Omit<CanvasItem, 'zIndex'> = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        src,
        name,
        x: canvasCenter - baseWidth / 2,
        y: canvasCenter - baseHeight / 2,
        width: baseWidth,
        height: baseHeight,
        naturalWidth,
        naturalHeight,
      };

      let nextItem: CanvasItem | null = null;
      setItems((prev) => {
        const maxZ = prev.reduce((acc, item) => Math.max(acc, item.zIndex), 0);
        nextItem = { ...itemTemplate, zIndex: maxZ + 1 };
        return [...prev, nextItem];
      });

      if (nextItem) {
        setSelectedId(nextItem.id);
      }
    },
    [setItems, setSelectedId],
  );

  const handleBackgroundMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === contentRef.current) {
      setSelectedId(null);
    }
  };

  const handleItemMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>, id: string) => {
      event.stopPropagation();
      event.preventDefault();
      if (!canvasRef.current) {
        return;
      }

      const item = items.find((entry) => entry.id === id);
      if (!item) {
        return;
      }

      const bounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - bounds.left) / zoomRef.current;
      const pointerY = (event.clientY - bounds.top) / zoomRef.current;

      dragState.current = {
        id,
        offsetX: pointerX - item.x,
        offsetY: pointerY - item.y,
      };
      setSelectedId(id);
    },
    [items],
  );

  const handleResizeMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>, id: string) => {
      event.stopPropagation();
      event.preventDefault();
      if (!canvasRef.current) {
        return;
      }

      const item = items.find((entry) => entry.id === id);
      if (!item) {
        return;
      }

      const bounds = canvasRef.current.getBoundingClientRect();
      const pointerX = (event.clientX - bounds.left) / zoomRef.current;
      const pointerY = (event.clientY - bounds.top) / zoomRef.current;

      resizeState.current = {
        id,
        originX: pointerX,
        originY: pointerY,
        startWidth: item.width,
        startHeight: item.height,
      };
      setSelectedId(id);
    },
    [items],
  );

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) {
      return;
    }

    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        continue;
      }
      const src = URL.createObjectURL(file);

      try {
        const image = await loadImageElement(src);
        const naturalWidth = image.naturalWidth || image.width;
        const naturalHeight = image.naturalHeight || image.height;

        if (!naturalWidth || !naturalHeight) {
          URL.revokeObjectURL(src);
          continue;
        }

        createdUrls.current.add(src);
        addCanvasItem(src, file.name, naturalWidth, naturalHeight);
      } catch {
        URL.revokeObjectURL(src);
      }
    }

    event.target.value = '';
  };

  const handleAddGeneratedImage = useCallback(
    async (imageUrl: string) => {
      let objectUrl: string | null = null;
      try {
        const response = await fetch(imageUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch generated image: ${response.status}`);
        }
        const blob = await response.blob();
        if (!blob.type.startsWith('image/')) {
          throw new Error('Generated asset is not an image.');
        }

        objectUrl = URL.createObjectURL(blob);
        const image = await loadImageElement(objectUrl);
        const naturalWidth = image.naturalWidth || image.width;
        const naturalHeight = image.naturalHeight || image.height;

        if (!naturalWidth || !naturalHeight) {
          throw new Error('Invalid image dimensions.');
        }

        createdUrls.current.add(objectUrl);
        addCanvasItem(objectUrl, `generated-${Date.now()}.png`, naturalWidth, naturalHeight);
      } catch (error) {
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
        }
        console.error('Failed to add generated image to canvas:', error);
        throw error;
      }
    },
    [addCanvasItem],
  );

  useEffect(() => {
    const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
    if (!pendingRaw) {
      return;
    }
    sessionStorage.removeItem('pendingCanvasItems');

    let pendingItems: { dataUrl: string; name?: string }[] = [];
    try {
      pendingItems = JSON.parse(pendingRaw);
    } catch (error) {
      console.error('Failed to parse pending canvas items:', error);
      return;
    }

    const importPending = async () => {
      for (const item of pendingItems) {
        try {
          const blob = dataUrlToBlob(item.dataUrl);
          const objectUrl = URL.createObjectURL(blob);
          const image = await loadImageElement(objectUrl);
          const naturalWidth = image.naturalWidth || image.width;
          const naturalHeight = image.naturalHeight || image.height;
          if (!naturalWidth || !naturalHeight) {
            throw new Error('Stored image has invalid dimensions.');
          }
          createdUrls.current.add(objectUrl);
          addCanvasItem(objectUrl, item.name || `generated-${Date.now()}.png`, naturalWidth, naturalHeight);
        } catch (error) {
          console.error('Failed to import stored canvas image:', error);
        }
      }
    };

    importPending();
  }, [addCanvasItem]);

  const handleDeleteSelected = () => {
    if (!selectedId) {
      return;
    }
    setItems((prev) => {
      const target = prev.find((item) => item.id === selectedId);
      if (target) {
        createdUrls.current.delete(target.src);
        URL.revokeObjectURL(target.src);
      }
      return prev.filter((item) => item.id !== selectedId);
    });
    setSelectedId(null);
  };

  const handleBringForward = () => {
    if (!selectedId) {
      return;
    }
    setItems((prev) => {
      const maxZ = prev.reduce((acc, item) => Math.max(acc, item.zIndex), 0);
      return prev.map((item) =>
        item.id === selectedId ? { ...item, zIndex: maxZ + 1 } : item,
      );
    });
  };

  const handleSendBackward = () => {
    if (!selectedId) {
      return;
    }
    setItems((prev) => {
      const minZ = prev.reduce((acc, item) => Math.min(acc, item.zIndex), Infinity);
      return prev.map((item) =>
        item.id === selectedId ? { ...item, zIndex: minZ - 1 } : item,
      );
    });
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(MAX_ZOOM, roundToTwo(prev + ZOOM_STEP)));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(MIN_ZOOM, roundToTwo(prev - ZOOM_STEP)));
  };

  const handleResetView = () => {
    setZoom(1);
    const container = containerRef.current;
    if (container) {
      container.scrollLeft = (CANVAS_DIMENSION - container.clientWidth) / 2;
      container.scrollTop = (CANVAS_DIMENSION - container.clientHeight) / 2;
    }
  };

  const handleLayerSelect = (id: string) => {
    setSelectedId(id);
    const item = items.find((entry) => entry.id === id);
    if (!item || !containerRef.current) {
      return;
    }
    const container = containerRef.current;
    const viewWidth = container.clientWidth / zoom;
    const viewHeight = container.clientHeight / zoom;
    container.scrollLeft = (item.x + item.width / 2) * zoom - viewWidth / 2;
    container.scrollTop = (item.y + item.height / 2) * zoom - viewHeight / 2;
  };

  const handleExport = useCallback(async () => {
    if (!items.length || isExporting) {
      return;
    }

    setIsExporting(true);

    try {
      const minX = Math.min(...items.map((item) => item.x));
      const minY = Math.min(...items.map((item) => item.y));
      const maxX = Math.max(...items.map((item) => item.x + item.width));
      const maxY = Math.max(...items.map((item) => item.y + item.height));

      const padding = 32;
      const exportWidth = Math.ceil(maxX - minX + padding * 2);
      const exportHeight = Math.ceil(maxY - minY + padding * 2);

      if (exportWidth <= 0 || exportHeight <= 0) {
        return;
      }

      const canvasElement = document.createElement('canvas');
      canvasElement.width = exportWidth;
      canvasElement.height = exportHeight;
      const context = canvasElement.getContext('2d');

      if (!context) {
        throw new Error('无法创建画布上下文。');
      }

      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, exportWidth, exportHeight);

      const orderedItems = [...items].sort((a, b) => a.zIndex - b.zIndex);

      for (const item of orderedItems) {
        try {
          const image = await loadImageElement(item.src);
          const naturalWidth = item.naturalWidth || image.naturalWidth || image.width;
          const naturalHeight = item.naturalHeight || image.naturalHeight || image.height;

          if (!naturalWidth || !naturalHeight) {
            continue;
          }

          const scale = Math.min(item.width / naturalWidth, item.height / naturalHeight);
          const drawWidth = naturalWidth * scale;
          const drawHeight = naturalHeight * scale;
          const offsetX = padding + item.x - minX + (item.width - drawWidth) / 2;
          const offsetY = padding + item.y - minY + (item.height - drawHeight) / 2;

          context.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
        } catch {
          // Skip any layer that fails during export.
        }
      }

      const blob = await canvasToBlob(canvasElement);
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `canvas-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch {
      // Silently ignore export errors for now.
    } finally {
      setIsExporting(false);
    }
  }, [isExporting, items]);

  return (
    <Box sx={{ position: 'relative', height: 'calc(100vh - 112px)', width: '100%' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h1" fontSize="26px" fontWeight={600}>
            Designer Canvas
          </Typography>
          <Typography variant="body2" color="text.secondary">
            上传素材，拖拽排布，快速拼出你的灵感板。
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          缩放 {Math.round(zoom * 100)}%
        </Typography>
      </Stack>

      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          width: '100%',
          height: '100%',
          borderRadius: 2,
          overflow: 'auto',
          position: 'relative',
          background:
            'radial-gradient(circle at center, rgba(255,255,255,0.4) 0%, rgba(240,242,245,0.9) 100%)',
          boxShadow: 'inset 0 0 0 1px rgba(33,37,41,0.04)',
        }}
      >
        <Box
          ref={canvasRef}
          sx={{
            position: 'relative',
            width: CANVAS_DIMENSION,
            height: CANVAS_DIMENSION,
            margin: '0 auto',
            backgroundImage:
              'linear-gradient(0deg, rgba(0,0,0,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px)',
            backgroundSize: '64px 64px',
            backgroundColor: '#f6f7f9',
          }}
        >
          <Box
            ref={contentRef}
            onMouseDown={handleBackgroundMouseDown}
            sx={{
              position: 'absolute',
              inset: 0,
              transform: `scale(${zoom})`,
              transformOrigin: 'top left',
              pointerEvents: 'auto',
            }}
          >
            {sortedItems.map((item) => (
              <Box
                key={item.id}
                onMouseDown={(event) => handleItemMouseDown(event, item.id)}
                sx={{
                  position: 'absolute',
                  left: item.x,
                  top: item.y,
                  width: item.width,
                  height: item.height,
                  cursor: 'move',
                  zIndex: item.zIndex,
                  border: item.id === selectedId ? '2px solid #007BFF' : '1px solid rgba(33, 37, 41, 0.12)',
                  borderRadius: 1,
                  boxShadow: item.id === selectedId ? '0 0 0 4px rgba(0, 123, 255, 0.16)' : '0 4px 12px rgba(15,23,42,0.12)',
                  backgroundColor: '#fff',
                  overflow: 'hidden',
                  transition: 'border 0.15s ease, box-shadow 0.15s ease',
                  userSelect: 'none',
                }}
              >
                <Box
                  component="img"
                  src={item.src}
                  alt={item.name}
                  sx={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: 'block',
                    pointerEvents: 'none',
                    userSelect: 'none',
                  }}
                  draggable={false}
                />
                <Box
                  onMouseDown={(event) => handleResizeMouseDown(event, item.id)}
                  sx={{
                    position: 'absolute',
                    width: 16,
                    height: 16,
                    borderRadius: '4px',
                    bottom: 6,
                    right: 6,
                    backgroundColor: '#007BFF',
                    boxShadow: '0 2px 6px rgba(0, 123, 255, 0.45)',
                    cursor: 'nwse-resize',
                  }}
                />
              </Box>
            ))}
          </Box>
        </Box>
      </Box>

      <Paper
        elevation={6}
        sx={{
          position: 'fixed',
          left: { xs: 16, sm: 288 },
          right: 16,
          bottom: 24,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 1,
          padding: '12px 16px',
          borderRadius: 999,
      backdropFilter: 'blur(12px)',
      backgroundColor: 'rgba(255,255,255,0.88)',
      zIndex: 1200,
    }}
  >
        <Tooltip title="上传素材">
          <IconButton color="primary" onClick={triggerFileUpload}>
            <AddPhotoAlternateIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="AI 生图">
          <IconButton color="primary" onClick={() => setIsInferenceOpen(true)}>
            <AutoAwesomeIcon />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
        <Tooltip title="置于最上层">
          <span>
            <IconButton
              onClick={handleBringForward}
              disabled={!selectedId}
              color="primary"
            >
              <FlipToFrontIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="置于最底层">
          <span>
            <IconButton
              onClick={handleSendBackward}
              disabled={!selectedId}
              color="primary"
            >
              <FlipToBackIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="删除素材">
          <span>
            <IconButton
              onClick={handleDeleteSelected}
              disabled={!selectedId}
              color="secondary"
            >
              <DeleteOutlineIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={isExporting ? '导出中...' : '导出PNG'}>
          <span>
            <IconButton
              onClick={handleExport}
              disabled={isExporting || !items.length}
              color="primary"
            >
              <DownloadIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
        <Tooltip title="缩小">
          <IconButton onClick={handleZoomOut}>
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="放大">
          <IconButton onClick={handleZoomIn}>
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="回到中心">
          <IconButton onClick={handleResetView}>
            <CenterFocusStrongIcon />
          </IconButton>
        </Tooltip>
      </Paper>

      <Paper
        elevation={4}
        sx={{
          position: 'fixed',
          top: 120,
          right: 24,
          width: 240,
        maxHeight: '60vh',
        overflow: 'auto',
        borderRadius: 3,
        backgroundColor: 'rgba(255,255,255,0.92)',
        zIndex: 1100,
      }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle2" color="text.secondary">
            图层
          </Typography>
        </Box>
        <Divider />
        {sortedItems.length ? (
          <List dense disablePadding>
            {[...sortedItems].reverse().map((item, index) => (
              <ListItemButton
                key={item.id}
                selected={item.id === selectedId}
                onClick={() => handleLayerSelect(item.id)}
                sx={{ py: 1 }}
              >
                <ListItemText
                  primaryTypographyProps={{
                    noWrap: true,
                    fontSize: 13,
                    fontWeight: item.id === selectedId ? 600 : 500,
                  }}
                  primary={`${sortedItems.length - index}. ${item.name}`}
                />
              </ListItemButton>
            ))}
          </List>
        ) : (
          <Box sx={{ px: 2, py: 2 }}>
            <Typography variant="body2" color="text.secondary">
              暂无素材，点击下方工具栏上传。
            </Typography>
          </Box>
        )}
      </Paper>

      <Dialog
        open={isInferenceOpen}
        onClose={() => setIsInferenceOpen(false)}
        fullWidth
        maxWidth="lg"
      >
        <DialogTitle
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            pr: 2,
          }}
        >
          AI 生图
          <IconButton onClick={() => setIsInferenceOpen(false)}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          <InferencePanel onAddToCanvas={handleAddGeneratedImage} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsInferenceOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={handleFilesSelected}
      />
    </Box>
  );
};

function clamp(value: number, min: number, max: number) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function roundToTwo(value: number) {
  return Math.round(value * 100) / 100;
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('导出失败'));
      }
    }, 'image/png');
  });
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load image.'));
    image.src = src;
  });
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [metadata, base64Data] = dataUrl.split(',');
  if (!metadata || !base64Data) {
    throw new Error('Invalid data URL.');
  }
  const match = metadata.match(/data:(.*?);base64/);
  const mime = match?.[1] || 'image/png';
  const binary = atob(base64Data);
  const length = binary.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

export default CanvasPage;
