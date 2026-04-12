BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

ALLOWED_TAGS = [
    "small_object", "multi_object_scene", "fine_grained_attribute",
    "spatial_relation", "visible_text", "occlusion", "cluttered_background",
    "low_contrast", "similar_objects", "dominant_distractor",
    "attribute_ambiguity",
]

# Fixed attribute axes — enforces LPCV-style discrimination.
# Derived from analysis of the official 222-text corpus.
ATTRIBUTE_AXES = [
    "color", "size", "state", "position", "material",
    "shape", "visible_text", "distinctive_part",
]

SYSTEM_PROMPT = """You are an expert computer vision dataset annotator for fine-grained image-text retrieval (LPCV 2026 Track 1).

TASK: Analyze ONE image and generate structured object-centric retrieval annotations that mimic the official corpus style — ultra-short referring expressions such as:
  "the white ping pong ball", "the tilted monitor", "the round mouse",
  "the airplane with red tail", "the pig facing left",
  "the rabbit with one ear up",
  "the ping pong ball with black text", "the keyboard with white key labels",
  "tent with white lettering", "the logo says up north".

GOAL: Train a model that can:
- recognize both dominant AND small secondary objects (mice, keyboards, logos, accessories),
- distinguish fine-grained attributes on the CORRECT attribute axis (color vs state vs size vs shape),
- separate visibly distinguishable instances of the same category,
- reject plausible but indisputably false hard negatives.

Each "object" = one visually retrievable INSTANCE, not one semantic class. Multiple same-category instances that are distinguishable (color, position, state, size, text presence) MUST be split into separate object blocks.

CRITICAL RULES
1. STRICT VISUAL EVIDENCE. Never hallucinate.
2. TEXT AWARENESS NOT OCR. When an object has visible text, lettering, or a logo, describe its PRESENCE using patterns like:
   - "with [color] text"  →  "the ball with black text", "the hoodie with white lettering"
   - "with no text"       →  "the ping pong ball with no text"  (useful when other instances DO have text)
   - "the logo says [X]"  →  ONLY for short (≤4 words), clearly legible, prominent logos or slogans.
   NEVER attempt to transcribe multi-word body text or guess partially-visible characters.
3. INSTANCE SEPARATION. Distinguishable same-category instances → separate blocks. Each block's texts must uniquely identify THAT instance (via color, position, state, text presence, or legible logo).
4. PLAUSIBLE YET FALSE NEGATIVES. Every hard negative must pass BOTH checks:
   (a) FALSE for the entire image — not even partially true anywhere in the scene.
   (b) CATEGORY-PLAUSIBLE — real instances of this object type CAN have this attribute
       in the real world. Never negate a defining species/category trait:
       ✗ "solid black zebra"   — zebras are always striped (category-impossible)
       ✗ "wingless butterfly"  — contradicts the category definition
       ✗ "the square monitor"  — consumer monitors are never square
       ✗ "the pink elephant"   — elephants are never pink
       ✓ "the zebra facing right"  — plausible (some do), false here if facing left
       ✓ "the adult bear cub"  — size/age confusion a model might make
       ✓ "the white keyboard"  — perfectly plausible, false if keyboard is black
   Hard negatives must represent mistakes a vision model could realistically make, not
   obvious impossibilities.
5. SECONDARY OBJECT COVERAGE. You MUST include small but identifiable secondary objects (keyboard, mouse, cable, sticker, logo, sign, label, background animal). Failure mode being fixed: small objects get dominated by large objects.
6. ATTRIBUTE-AXIS DIVERSITY. Across ALL positives for one image, cover ≥4 of these axes: {color, size, state, position, material, shape, visible_text, distinctive_part}. Each object block should use ≥2 different axes when possible.
7. CONCISE NOUN PHRASES. 3–10 words. Lowercase.
8. No duplicates, no near-duplicates.
9. All texts in one object block refer to the exact same instance.

PER OBJECT, GENERATE
- positive_texts (1-2): highly accurate discriminative descriptions. Tag each with its attribute_axis.
- weak_positives (0-1): correct but less specific.
- hard_negatives_attribute (1-2): ONE indisputably wrong INTRINSIC fact (color/shape/state/size/text flip). Edit-type: "axis_flip" or "axis_swap". Must pass the plausibility check from Rule 4.
- hard_negatives_scene (0-2): object identical, ONE surrounding/context fact altered (support surface, relative position to another clearly-present object).
- relational_texts (0-2): true relations using explicit, unambiguous relative position to another visible object.
- text_on_object (0-1): If the object has visible text, lettering, or a logo, produce ONE description using the text-awareness patterns from Rule 2. Use attribute_axis "visible_text". Leave empty if no text is visible on the object.

NEGATIVE DESIGN GUIDANCE (bad-case-driven)
- Before emitting each negative, run TWO silent checks:
  1. "Is this false for the whole image?" — if no/unsure → discard.
  2. "Can a real [object type] actually have this attribute?" — if no → discard.
- Target axes where models make real errors: exact color (white/grey/silver/black confusion),
  orientation/pose (facing left vs right), state (on/off, open/closed, tilted/upright),
  size rank (largest vs smallest among similar objects), text presence (with/without text).

OUTPUT: valid JSON matching the provided schema. salience ∈ {primary, secondary}. challenge_tags from the fixed list only.
"""

USER_PROMPT = """Analyze this image and generate structured image-to-text retrieval training data.

Requirements:
- Cover ≥3 distinct object categories when present; include small secondary objects (keyboard, mouse, logo, label, cable, accessory).
- Split visibly distinguishable same-category instances into separate object blocks.
- Across all positives in this image, use ≥4 different attribute axes from {color, size, state, position, material, shape, visible_text, distinctive_part}.
- Each positive_text item must declare its attribute_axis.
- For text_on_object: describe the PRESENCE of text/lettering using patterns like "with [color] text", "with white lettering", "with no text", or "the logo says [X]" (≤4 words, clearly legible only). Do NOT transcribe multi-word text verbatim.
- Hard negatives must be absolutely false for the WHOLE image.
- Concise noun phrases, 3–10 words, lowercase.
- Return valid JSON only.
"""

FEW_SHOT_EXAMPLES = [
    {
        "image_id": "example_ping_pong_balls.jpg",
        "global_texts": ["three ping pong balls on a dark scratched surface"],
        "challenge_tags": ["multi_object_scene", "fine_grained_attribute",
                           "spatial_relation", "similar_objects", "visible_text"],
        "objects": [
            {"object_name": "ping pong ball", "salience": "primary",
             "positive_texts": [
                 {"text": "the white ping pong ball", "attribute_axis": "color"},
                 {"text": "the ball at the upper left", "attribute_axis": "position"}],
             "weak_positives": [],
             "hard_negatives_attribute": [
                 {"text": "the blue ping pong ball", "edit_type": "axis_flip",
                  "rationale": "no blue ball present"},
                 {"text": "the cube-shaped ping pong ball", "edit_type": "axis_swap",
                  "rationale": "all balls are round"}],
             "hard_negatives_scene": [
                 {"text": "the white ping pong ball on a wooden table",
                  "rationale": "surface is scratched dark metal, not wood"}],
             "relational_texts": ["the white ping pong ball left of the orange ball"],
             "text_on_object": ["the ping pong ball with black text"]},
            {"object_name": "ping pong ball", "salience": "primary",
             "positive_texts": [
                 {"text": "the orange ping pong ball", "attribute_axis": "color"},
                 {"text": "the ball at the upper right", "attribute_axis": "position"}],
             "weak_positives": [],
             "hard_negatives_attribute": [
                 {"text": "the purple ping pong ball", "edit_type": "axis_flip",
                  "rationale": "no purple ball present"}],
             "hard_negatives_scene": [],
             "relational_texts": ["the orange ping pong ball right of the white ball"],
             "text_on_object": ["the ping pong ball with no text"]},
            {"object_name": "ping pong ball", "salience": "primary",
             "positive_texts": [
                 {"text": "the light green ping pong ball", "attribute_axis": "color"},
                 {"text": "the bottom ping pong ball", "attribute_axis": "position"}],
             "weak_positives": [],
             "hard_negatives_attribute": [
                 {"text": "the black ping pong ball", "edit_type": "axis_flip",
                  "rationale": "no black ball present"}],
             "hard_negatives_scene": [],
             "relational_texts": ["the green ping pong ball below the white ball"],
             "text_on_object": ["the ping pong ball with blue text"]},
        ],
    },
    {
        "image_id": "example_desk_multi_monitor.jpg",
        "global_texts": ["a desk with three monitors, keyboard and mouse"],
        "challenge_tags": ["multi_object_scene", "small_object",
                           "similar_objects", "dominant_distractor"],
        "objects": [
            {"object_name": "monitor", "salience": "primary",
             "positive_texts": [
                 {"text": "the tilted monitor", "attribute_axis": "state"},
                 {"text": "the black crt monitor", "attribute_axis": "color"}],
             "weak_positives": ["crt monitor"],
             "hard_negatives_attribute": [
                 {"text": "the white monitor", "edit_type": "axis_flip",
                  "rationale": "all monitors are black"},
                 {"text": "the glowing monitor", "edit_type": "axis_swap",
                  "rationale": "all monitors are off"}],
             "hard_negatives_scene": [
                 {"text": "monitor mounted on the wall",
                  "rationale": "monitors stand on the desk"}],
             "relational_texts": ["the monitor behind the keyboard"],
             "text_on_object": ["the monitor with a white brand logo"]},
            {"object_name": "keyboard", "salience": "secondary",
             "positive_texts": [
                 {"text": "the black keyboard", "attribute_axis": "color"},
                 {"text": "the rectangular keyboard", "attribute_axis": "shape"}],
             "weak_positives": ["wired keyboard"],
             "hard_negatives_attribute": [
                 {"text": "the white keyboard", "edit_type": "axis_flip",
                  "rationale": "keyboard is black"},
                 {"text": "the rounded keyboard", "edit_type": "axis_swap",
                  "rationale": "keyboard is rectangular not rounded"}],
             "hard_negatives_scene": [],
             "relational_texts": ["the keyboard in front of the monitors"],
             "text_on_object": ["the keyboard with white key labels"]},
            {"object_name": "mouse", "salience": "secondary",
             "positive_texts": [
                 {"text": "the round mouse", "attribute_axis": "shape"},
                 {"text": "the black mouse on the mousepad", "attribute_axis": "color"}],
             "weak_positives": ["wired mouse"],
             "hard_negatives_attribute": [
                 {"text": "the white mouse", "edit_type": "axis_flip",
                  "rationale": "mouse is black"},
                 {"text": "the rectangular mouse", "edit_type": "axis_swap",
                  "rationale": "mouse is rounded"}],
             "hard_negatives_scene": [],
             "relational_texts": ["the mouse right of the keyboard"],
             "text_on_object": []},
        ],
    },
]

RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "image_id": {"type": "string"},
        "global_texts": {"type": "array", "items": {"type": "string"}},
        "challenge_tags": {"type": "array",
            "items": {"type": "string", "enum": ALLOWED_TAGS}},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "object_name": {"type": "string"},
                "salience": {"type": "string", "enum": ["primary", "secondary"]},
                "positive_texts": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "attribute_axis": {"type": "string", "enum": ATTRIBUTE_AXES},
                    }, "required": ["text", "attribute_axis"]}},
                "weak_positives": {"type": "array", "items": {"type": "string"}},
                "hard_negatives_attribute": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "edit_type": {"type": "string",
                            "enum": ["axis_flip", "axis_swap", "cross_object", "sibling"]},
                        "rationale": {"type": "string"},
                    }, "required": ["text", "edit_type", "rationale"]}},
                "hard_negatives_scene": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "rationale": {"type": "string"},
                    }, "required": ["text", "rationale"]}},
                "relational_texts": {"type": "array", "items": {"type": "string"}},
                "text_on_object": {"type": "array", "items": {"type": "string"},
                    "description": "Text-awareness descriptions: 'with [color] text', "
                                   "'with white lettering', 'with no text', or "
                                   "'the logo says [X]' (≤4 words, clearly legible only)."},
            },
            "required": ["object_name", "salience", "positive_texts",
                "weak_positives", "hard_negatives_attribute",
                "hard_negatives_scene", "relational_texts", "text_on_object"],
        }},
    },
    "required": ["image_id", "global_texts", "challenge_tags", "objects"],
}

VERIFY_PROMPT = """You are a strict visual fact-checker. Decide whether ANY region, instance, part, accessory, or attribute in the image could be truthfully described by the expression.

Answer "yes" if ANY part matches, even partially.
Answer "no" ONLY if confident no region matches.
When uncertain, answer "yes" (we prefer dropping borderline negatives).

Expression: "{TEXT}"

Return JSON: {{"match": "yes"|"no"|"unsure", "reason": "<short>", "region": "<which part or none>"}}
"""

BATCH_VERIFY_PROMPT = """You are a strict visual fact-checker. Given an image and a JSON list of text descriptions, decide for each whether ANY region, instance, part, accessory, or attribute in the image could be truthfully described by that text.

For each item return:
- "no"     — you are confident NO part of the image matches
- "yes"    — any part matches, even partially or approximately
- "unsure" — borderline; when in doubt prefer "yes"

We keep only hard negatives that are indisputably false. Answer "no" only when clearly certain.

The output "results" array MUST have the same length and order as the input list.
"""

BATCH_VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "text":  {"type": "string"},
                    "match": {"type": "string", "enum": ["yes", "no", "unsure"]},
                },
                "required": ["text", "match"],
            },
        },
    },
    "required": ["results"],
}

