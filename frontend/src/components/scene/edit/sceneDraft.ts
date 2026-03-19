export type SceneMaterialDraftSlot = {
  id: string;
  label: string;
  layer_type: 'background' | 'subject' | 'decor' | string;
  prompt: string;
};

export type SceneGraphObject = {
  id: string;
  label: string;
  layer_type: 'background' | 'subject' | 'decor' | string;
  bbox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  depth_order: number;
  prompt: string;
  needs_transparent_bg: boolean;
};

export type SceneReferenceDraft = {
  provider: 'gemini' | 'siliconflow' | string;
  image_base64: string;
  scene_graph: {
    summary: string;
    text_safe_zone: {
      x: number;
      y: number;
      w: number;
      h: number;
    };
    objects: SceneGraphObject[];
  };
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
  reference: SceneReferenceDraft;
};
