from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline

# Define the pre-trained Pegasus model name. "google/pegasus-xsum" is used here for summarization.
model_name = "google/pegasus-xsum"

# Load the Pegasus tokenizer and model using the specified model name
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Input text to be summarized
test_text = (
    "Coming from a Corsican family, Bonaparte rose through the ranks of the army during the French Revolution. "
    "He proved to be a first-rate military talent. His campaigns in Italy and Egypt, in particular, made him popular, "
    "enabling him to seize power in France through the coup d'état of 18 Brumaire VIII (9 November 1799), initially as one of three consuls. "

    "From 1799 to 1804, as First Consul of the French Republic, and then until 1814 and again in 1815 as Emperor of the French, "
    "he presided over a dictatorial regime with plebiscitary elements. Through various reforms—such as those of the judiciary through the Code Civil "
    "or administrative reforms—Napoleon shaped the state structures of France up to the present day and initiated the creation of modern civil law "
    "in occupied European states. "

    "In foreign policy, with the support of his army, he temporarily gained control over large parts of continental Europe. "
    "He was also King of Italy from 1805 and Protector of the Confederation of the Rhine from 1806 to 1813. "
    "Additionally, he installed family members and confidants as monarchs in several other states. "

    "When he initiated the dissolution of the Holy Roman Empire in 1806, the state structure of Central Europe became a central issue "
    "in the 19th century. While he initially spread the idea of the nation-state outside of France, the success of this idea made it more "
    "difficult to maintain the Napoleonic order in Europe, particularly in Spain, Germany, and ultimately Russia. "

    "The disastrous outcome of the campaign against Russia in 1812 was followed by the Wars of Liberation, shaking France's dominance "
    "in large parts of Europe and ultimately leading to Napoleon's fall. After a brief period of exile on Elba, he returned to power for a "
    "hundred days in 1815. Following his defeat at the Battle of Waterloo, he was exiled to the island of St. Helena for the rest of his life."
)

# Tokenize the input text, preparing it for input into the Pegasus model
# - Truncation is enabled to prevent overly long inputs
# - Padding is set to "longest" to match tensor dimensions across samples if needed
# - Return tensor format is set to "pt" (PyTorch tensors)
tokens = pegasus_tokenizer(test_text, truncation=True, padding="longest", return_tensors="pt")

# Generate a summary using the Pegasus model directly
# - `max_length=250`: Sets maximum length for the summary text
# - `min_length=150`: Ensures the summary has enough content
# - `length_penalty=2.0`: Controls length preference, encouraging shorter output
# - `num_beams=4`: Specifies the beam search count for generating multiple summary paths, improving coherence
encoded_summary = pegasus_model.generate(
    **tokens,
    max_length=250,
    min_length=150,
    length_penalty=2.0,
    num_beams=4
)

# Decode the generated tokenized summary into readable text, skipping special tokens like [PAD]
decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)

# Display the direct model-generated summary
print("Direct Model Summary:", decoded_summary)

# Initialize the summarization pipeline using the same model and tokenizer
# - The pipeline simplifies the summarization process, allowing min/max length specifications
summarizer = pipeline("summarization", model=pegasus_model, tokenizer=pegasus_tokenizer, framework="pt")

# Generate a summary through the pipeline with adjusted parameters
# - `min_length=150` and `max_length=300` provide control over the output size
# - `length_penalty=2.0` encourages brevity while allowing sufficient length for meaningful content
pipeline_summary = summarizer(test_text, min_length=150, max_length=300, length_penalty=2.0)[0]["summary_text"]

# Display the summary generated using the pipeline
print("Pipeline Summary:", pipeline_summary)
from unittest import TestCase


