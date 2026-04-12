BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ALLOWED_TAGS = [
    "small_object",
    "multi_object_scene",
    "fine_grained_attribute",
    "spatial_relation",
    "ocr",
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
- handle multi-object scenes (especially distinguishing similar instances),
- use relational cues when useful,
- use OCR cues when fully legible,
- learn hard negatives that are close, plausible, but unequivocally false.

Each "object" means one visually retrievable object instance in the image, not just one semantic category.
Multiple objects may belong to the same category if they are visibly distinguishable.
For example, three ping pong balls of different colors or in different positions should be annotated as three separate object blocks if each one can be reliably identified.
Objects may also come from different categories.

Critical rules:
1. STRICT VISUAL EVIDENCE: Only generate annotations supported by absolute clear visual evidence. Do not hallucinate or infer missing parts.
2. NO OCR GUESSING: If text or numbers are blurry, pixelated, or partially obscured, DO NOT extract them. Use OCR only if it is completely legible to the naked eye.
3. INSTANCE SEPARATION AND DISAMBIGUATION: If multiple objects of the same category appear in the image, and they are visibly distinguishable by color, size, position, state, visible part, or legible text, treat them as separate target objects and create separate object blocks for them. The positive and relational texts for each block must uniquely identify that exact instance using explicit spatial anchors, unique visible attributes, or legible OCR when available. Do not merge multiple distinguishable instances of the same category into one object block.
4. UNAMBIGUOUS NEGATIVES: Hard negatives must contain an absolute, visually indisputable contradiction. Avoid subjective or borderline attributes (e.g., do not use "blue" as a negative for a "bluish-grey" object).
5. Include both dominant objects and smaller secondary objects if they could plausibly serve as retrieval targets. Ignore objects that are too tiny or ambiguous.
6. Keep all prompts concise, natural, and similar to real-world human search queries (**typically 3-10 words**, noun phrases preferred).
7. Do not over-describe background details or invent hidden materials, brands, or intent.
8. Lowercase all generated phrases unless capitalization is absolutely necessary for exact OCR matching.
9. Do not generate duplicates or near-duplicates.
10. All texts inside one object block must refer to the exact same target object instance.
11. A negative must be false for the full image, not merely false for the target region.

For each target object, generate:
- positive_texts:
  1-2 highly accurate, discriminative descriptions of the target object.
- weak_positives:
  0-1 correct but less specific descriptions of the same target object.
- hard_negatives_attribute:
  0-2 descriptions containing exactly ONE indisputably wrong intrinsic fact (e.g., a drastically different color, shape, or state).
- hard_negatives_scene:
  0-2 descriptions where the object is identical, but ONE surrounding-scene or local-context fact is completely altered (e.g., changing the support surface or relative position).
- relational_texts:
  0-2 true descriptions based on explicit, unambiguous relative position or interaction with another clearly visible object.
- ocr_texts:
  0-1 true descriptions based STRICTLY on fully legible text/logos. If you have to squint, guess, or infer the letters due to blur, folds, or occlusion, DO NOT extract it.

Output requirements:
- Return valid JSON only.
- Top-level output must be object-centric.
- If a field does not apply or cannot be generated safely, return an empty list []. Do not force generation.
- Salience must be "primary" or "secondary".
- Challenge tags must be chosen only from the provided list.
"""

USER_PROMPT = """Analyze this image and generate structured image to text retrieval training data.

Requirements:
- Include all reasonably identifiable retrieval targets, including smaller secondary objects.
- Treat visibly distinguishable instances of the same category as separate objects.
- For example, multiple ping pong balls, cups, bottles, or signs should be annotated separately if they can be reliably told apart by visible attributes or position.
- Prefer concise noun phrases.
- Use only visible, image-grounded attributes.
- Generate strong positives, weak positives, attribute hard negatives, scene hard negatives, relational texts, and OCR texts when supported.
- Return valid JSON only following the required schema.
"""

FEW_SHOT_EXAMPLE = [
    {
        "image_id": "example_ping_pong_balls.jpg",
        "global_texts": [
            "three ping pong balls on a dark scratched surface",
            "colored table tennis balls grouped together",
        ],
        "challenge_tags": [
            "multi_object_scene",
            "fine_grained_attribute",
            "spatial_relation",
            "similar_objects",
        ],
        "objects": [
            {
                "object_name": "ping pong ball",
                "salience": "primary",
                "positive_texts": [
                    "the white ping pong ball",
                    "the white ball at the upper left",
                ],
                "weak_positives": [],
                "hard_negatives_attribute": ["the blue ping pong ball"],
                "hard_negatives_scene": [
                    "the white ping pong ball on a wooden table",
                    "the white ping pong ball below the green ball",
                ],
                "relational_texts": [
                    "the white ping pong ball left of the orange ball",
                    "the white ping pong ball above the green ball",
                ],
                "ocr_texts": [],
            },
            {
                "object_name": "ping pong ball",
                "salience": "primary",
                "positive_texts": [
                    "the orange ping pong ball",
                    "the orange ball at the upper right",
                ],
                "weak_positives": [],
                "hard_negatives_attribute": ["the purple ping pong ball"],
                "hard_negatives_scene": [
                    "the orange ping pong ball on a wooden table",
                    "the orange ping pong ball left of the white ball",
                ],
                "relational_texts": [
                    "the orange ping pong ball right of the white ball",
                    "the orange ping pong ball above the green ball",
                ],
                "ocr_texts": [],
            },
            {
                "object_name": "ping pong ball",
                "salience": "primary",
                "positive_texts": [
                    "the light green ping pong ball",
                    "the bottom ping pong ball",
                ],
                "weak_positives": [],
                "hard_negatives_attribute": ["the black ping pong ball"],
                "hard_negatives_scene": [
                    "the green ping pong ball on a wooden table",
                    "the green ping pong ball above the white ball",
                ],
                "relational_texts": [
                    "the green ping pong ball below the white ball",
                    "the green ping pong ball below the orange ball",
                ],
                "ocr_texts": [],
            },
        ],
    },
    {
        "image_id": "example_dual_computers.jpg",
        "global_texts": [
            "two desktop computer setups on a table",
            "crt monitors on desktop towers with keyboards and a mouse",
        ],
        "challenge_tags": [
            "multi_object_scene",
            "similar_objects",
            "spatial_relation",
            "ocr",
        ],
        "objects": [
            {
                "object_name": "monitor",
                "salience": "primary",
                "positive_texts": [
                    "black crt monitor with dell logo",
                    "crt monitor above desktop tower",
                ],
                "weak_positives": ["crt monitor"],
                "hard_negatives_attribute": ["white crt monitor"],
                "hard_negatives_scene": [
                    "crt monitor mounted on the wall",
                    "crt monitor on the floor",
                ],
                "relational_texts": [
                    "crt monitor behind keyboard",
                    "crt monitor above computer tower",
                ],
                "ocr_texts": ["crt monitor with dell logo"],
            },
            {
                "object_name": "desktop tower",
                "salience": "primary",
                "positive_texts": [
                    "black desktop tower with dell logo",
                    "desktop tower under crt monitor",
                ],
                "weak_positives": ["computer tower"],
                "hard_negatives_attribute": ["white computer tower"],
                "hard_negatives_scene": [
                    "computer tower under the table",
                    "computer tower stacked on another tower",
                ],
                "relational_texts": [
                    "desktop tower below monitor",
                    "desktop tower behind keyboard",
                ],
                "ocr_texts": ["desktop tower with dell logo"],
            },
            {
                "object_name": "keyboard",
                "salience": "secondary",
                "positive_texts": [
                    "black keyboard at the front of the table",
                    "keyboard in front of desktop tower",
                ],
                "weak_positives": ["black keyboard"],
                "hard_negatives_attribute": ["white keyboard"],
                "hard_negatives_scene": [
                    "keyboard on top of the monitor",
                    "keyboard under the table",
                ],
                "relational_texts": [
                    "keyboard below monitor",
                    "keyboard beside the mouse",
                ],
                "ocr_texts": ["keyboard with dell text"],
            },
            {
                "object_name": "mouse",
                "salience": "secondary",
                "positive_texts": [
                    "black wired mouse in the center",
                    "mouse between the keyboards",
                ],
                "weak_positives": ["wired mouse"],
                "hard_negatives_attribute": ["white mouse"],
                "hard_negatives_scene": [
                    "mouse without a cable",
                    "mouse on top of a keyboard",
                ],
                "relational_texts": [
                    "mouse below the monitors",
                    "mouse between the keyboards",
                ],
                "ocr_texts": ["mouse with dell logo"],
            },
        ],
    },
]

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
                    "ocr_texts",
                ],
            },
        },
    },
    "required": ["image_id", "global_texts", "challenge_tags", "objects"],
}
