import React, { useCallback } from 'react';
import { Container, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import InferencePanel from './inference/InferencePanel';

const InferencePage: React.FC = () => {
  const navigate = useNavigate();

  const handleAddToCanvas = useCallback(
    async (imageUrl: string) => {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch generated image: ${response.status}`);
      }
      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) {
        throw new Error('Generated asset is not an image.');
      }

      const dataUrl = await blobToDataUrl(blob);
      const pendingRaw = sessionStorage.getItem('pendingCanvasItems');
      let pendingItems: { dataUrl: string; name: string }[] = [];
      if (pendingRaw) {
        try {
          pendingItems = JSON.parse(pendingRaw);
        } catch (error) {
          console.error('Failed to parse pending canvas queue. Resetting.', error);
          pendingItems = [];
        }
      }
      pendingItems.push({
        dataUrl,
        name: `generated-${Date.now()}.png`,
      });
      sessionStorage.setItem('pendingCanvasItems', JSON.stringify(pendingItems));
      navigate('/canvas');
    },
    [navigate],
  );

  return (
    <Container maxWidth={false}>
      <Typography variant="h4" component="h1" gutterBottom>
        Stable Diffusion Inference
      </Typography>
      <InferencePanel onAddToCanvas={handleAddToCanvas} />
    </Container>
  );
};

export default InferencePage;

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to read blob data.'));
      }
    };
    reader.onerror = () => reject(new Error('Failed to convert blob to data URL.'));
    reader.readAsDataURL(blob);
  });
}
