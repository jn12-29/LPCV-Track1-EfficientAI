PROMPTS = """Role: You are an expert computer vision dataset annotator for fine-grained image-text retrieval. 

Input: An image containing multiple similar objects, texts, or distinct interactions.

Task: Generate fine-grained text descriptions (Positive Prompts) and strictly corresponding deceptive descriptions (Hard Negative Prompts) based on the image content. 

Analyze the image and generate annotations based on the following 4 categories. 
CRITICAL RULE 1: ONLY generate a category if the image genuinely contains clear visual evidence for it. Skip the category entirely if it does not fit the visual context (Do NOT hallucinate).
CRITICAL RULE 2: You may generate MULTIPLE annotation blocks for the same category if there are multiple distinct target objects in the image.

1. Attribute-based: Focus on highly specific colors, textures, materials, or local poses of a distinct object.
   - Examples: "The denim shorts with flower pattern", "Tiger looking left".
2. Absence/Negation: Describe an object by explicitly mentioning a feature it clearly LACKS compared to normal expectations.
   - Examples: "The ping pong ball with no text", "The double decker bus without roof".
3. Relational & Interactive: Describe an object based on its explicit interaction or relative position to ANOTHER object. Skip if it's an isolated object.
   - Examples: "The giraffe is looking down at the gray car", "Drum with drumsticks on top".
4. OCR & Scene Text: Describe an object based on the readable text, letters, or logos printed on it. Skip if no readable text exists.
   - Examples: "The hoodie has the words duluth minnesota", "Tent with white lettering".

Length & Style Constraint:
Keep all prompts extremely concise, natural, and phrased like real-world human search queries (typically 3-10 words). Use short noun phrases rather than full, verbose descriptive sentences. DO NOT over-describe background details.

Prompt Generation Rules:
- For EACH target object, generate 2-3 synonymous Positive Prompts (paraphrases) that describe the exact same target using slightly different phrasing or valid synonyms.
- For EACH target object, generate 2 Hard Negative Prompts. A Hard Negative Prompt MUST look highly similar to the positives but contain EXACTLY ONE factual error (e.g., changing "green" to "red", mutating the OCR text, or flipping "without" to "with").

Output strictly in JSON format without markdown wrappers. Here is an output example:
{
  "annotations": [
    {
      "category": "Attribute-based",
      "positives": [
        "The blue soccer ball with wavy lines",
        "A blue football featuring wavy patterns"
      ],
      "hard_negatives": [
        "The red soccer ball with wavy lines",
        "The blue soccer ball with straight lines"
      ]
    },
    {
      "category": "Absence/Negation",
      "positives": [
        "The double decker bus without roof",
        "An open-top double decker bus"
      ],
      "hard_negatives": [
        "The double decker bus with closed roof",
        "The single decker bus without roof"
      ]
    },
    {
      "category": "OCR & Scene Text",
      "positives": [
        "The hoodie has the words duluth minnesota",
        "The sweatshirt with duluth minnesota text"
      ],
      "hard_negatives": [
        "The hoodie has the words up north",
        "The t-shirt has the words duluth minnesota"
      ]
    }
  ]
}
"""
