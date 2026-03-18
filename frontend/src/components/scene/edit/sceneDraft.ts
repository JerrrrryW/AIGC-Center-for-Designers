export type SceneMaterialDraftSlot = {
  id: string;
  label: string;
  layer_type: 'background' | 'subject' | 'decor' | string;
  prompt: string;
};

export type SceneDraft = {
  brief: {
    prompt: string;
    summary: string;
    aspect_ratio: string;
  };
  copy: {
    headline: string;
    subtitle: string;
    body: string;
    primary_color: string;
  };
  controls: {
    positive_prompt: string;
    negative_prompt: string;
    steps: number;
    cfg_scale: number;
    seed_locked: boolean;
  };
  materials: {
    background: SceneMaterialDraftSlot | null;
    subjects: SceneMaterialDraftSlot[];
    decors: SceneMaterialDraftSlot[];
    slots: SceneMaterialDraftSlot[];
  };
};
