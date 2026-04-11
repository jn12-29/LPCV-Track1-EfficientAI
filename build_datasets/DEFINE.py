BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ALLOWED_TAGS = [
    "small_object",
    "multi_object_scene",
    "fine_grained_attribute",
    "spatial_relation",
    "ocr",
    "absence",
    "occlusion",
    "cluttered_background",
    "low_contrast",
    "similar_objects",
    "dominant_distractor",
    "attribute_ambiguity",
]

SYSTEM_PROMPT = """You are an expert computer vision dataset annotator for fine-grained image-text retrieval.

Your task is to analyze ONE image and generate structured object-centric retrieval annotations.

Goal:
Create high-quality retrieval training data that helps a model:
- recognize both dominant objects and smaller secondary objects,
- distinguish fine-grained visible attributes,
- handle multi-object scenes,
- use relational cues when useful,
- use OCR cues when visible,
- learn hard negatives that are highly plausible confusions.

Critical rules:
1. Only generate annotations supported by clear visual evidence. Do not hallucinate.
2. Include both dominant objects and smaller secondary objects if they are visually identifiable and could plausibly serve as retrieval targets.
3. Ignore objects that are too tiny, too blurry, or too ambiguous to describe reliably.
4. Keep all prompts concise, natural, and similar to real-world human search queries.
5. Use short noun phrases whenever possible, typically 3-10 words.
6. Do not over-describe background details.
7. Do not invent hidden attributes, brands, materials, intent, or unreadable text.
8. Avoid subjective or speculative words such as beautiful, nice, modern, expensive, cute, unless visually explicit and retrieval-useful.
9. Lowercase all generated phrases unless capitalization is necessary for clearly readable text.
10. Do not generate duplicates or near-duplicates.

For each target object, generate:
- positive_texts:
  2-3 highly accurate, concise, retrieval-useful descriptions of the exact same target object.
- weak_positives:
  1-3 correct but less specific descriptions of the same target object.
- hard_negatives_attribute:
  2 descriptions that stay very close to the positive descriptions but contain exactly one key factual error about the same object, such as color, text, state, local part, or attribute.
- hard_negatives_scene:
  2 descriptions of other real objects in the same image that are plausible distractors for retrieval.
- relational_texts:
  0-2 descriptions based on explicit interaction or relative position, only if clearly supported by the image.
- absence_texts:
  0-2 descriptions based on a clearly visible missing feature, only if genuinely discriminative.
- ocr_texts:
  0-2 descriptions based on clearly readable text, letters, or logos, only if present and legible.

Visible evidence types you may use:
- attribute: color, texture, material appearance, local shape, local pose, state
- absence/negation: a clearly missing expected part or feature
- relation/interaction: explicit relative position or interaction with another object
- OCR/scene text: clearly readable text or logo

Negative generation rules:
- Attribute negatives should stay highly similar to the positive descriptions and differ by exactly one key fact.
- Scene negatives should refer to other real objects in the same image, not unrelated imaginary objects.
- Negatives must be plausible confusions, not absurd mismatches.

Output requirements:
- Return valid JSON only.
- Top-level output must be object-centric, not category-centric.
- If a field does not apply, return an empty list.
- Salience must be one of: \"primary\" or \"secondary\".
- Challenge tags must be chosen only from:
  [
    \"small_object\",
    \"multi_object_scene\",
    \"fine_grained_attribute\",
    \"spatial_relation\",
    \"ocr\",
    \"absence\",
    \"occlusion\",
    \"cluttered_background\",
    \"low_contrast\",
    \"similar_objects\",
    \"dominant_distractor\",
    \"attribute_ambiguity\"
  ]
"""

USER_PROMPT = """Analyze this image and generate structured retrieval training data.

Requirements:
- Include all reasonably identifiable retrieval targets, including smaller secondary objects.
- Prefer concise noun phrases.
- Use only visible, image-grounded attributes.
- Generate strong positives, weak positives, attribute hard negatives, scene hard negatives, relational texts, absence texts, and OCR texts when supported.
- Return valid JSON only following the required schema.
"""

FEW_SHOT_EXAMPLE = {
    "image_id": "example_001.jpg",
    "global_texts": [
        "a desk setup with a monitor, keyboard, and mouse",
        "an office desk with computer equipment",
    ],
    "challenge_tags": [
        "small_object",
        "multi_object_scene",
        "fine_grained_attribute",
        "dominant_distractor",
    ],
    "objects": [
        {
            "object_name": "monitor",
            "salience": "primary",
            "positive_texts": [
                "the white monitor",
                "the bezeled monitor",
                "the grey monitor",
            ],
            "weak_positives": [
                "the desk monitor",
                "the computer monitor",
            ],
            "hard_negatives_attribute": [
                "the black monitor",
                "the illuminated monitor",
            ],
            "hard_negatives_scene": [
                "the white keyboard",
                "the black and grey mouse",
            ],
            "relational_texts": [
                "the monitor behind the keyboard",
            ],
            "absence_texts": [
                "the monitor with no sticky note",
            ],
            "ocr_texts": [],
        },
        {
            "object_name": "mouse",
            "salience": "secondary",
            "positive_texts": [
                "the round mouse",
                "the black and grey mouse",
            ],
            "weak_positives": [
                "the desk mouse",
                "the mouse near the keyboard",
            ],
            "hard_negatives_attribute": [
                "the white mouse",
                "the silver mouse",
            ],
            "hard_negatives_scene": [
                "the off monitor",
                "the white keyboard",
            ],
            "relational_texts": [
                "the mouse in front of the monitor",
                "the mouse next to the keyboard",
            ],
            "absence_texts": [],
            "ocr_texts": [],
        },
    ],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_id": {"type": "string"},
        "global_texts": {"type": "array", "items": {"type": "string"}},
        "challenge_tags": {
            "type": "array",
            "items": {"type": "string", "enum": ALLOWED_TAGS},
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object_name": {"type": "string"},
                    "salience": {"type": "string", "enum": ["primary", "secondary"]},
                    "positive_texts": {"type": "array", "items": {"type": "string"}},
                    "weak_positives": {"type": "array", "items": {"type": "string"}},
                    "hard_negatives_attribute": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "hard_negatives_scene": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "relational_texts": {"type": "array", "items": {"type": "string"}},
                    "absence_texts": {"type": "array", "items": {"type": "string"}},
                    "ocr_texts": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "object_name",
                    "salience",
                    "positive_texts",
                    "weak_positives",
                    "hard_negatives_attribute",
                    "hard_negatives_scene",
                    "relational_texts",
                    "absence_texts",
                    "ocr_texts",
                ],
            },
        },
    },
    "required": ["image_id", "global_texts", "challenge_tags", "objects"],
}
