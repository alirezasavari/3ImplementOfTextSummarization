import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline


# Function to generate a summary using a pre-trained Pegasus model
def generate_summary_from_model(input_text, model_identifier, max_summary_length=250, min_summary_length=150,
                                penalty_for_length=2.0, beam_count=4):
    # Load the Pegasus tokenizer specific to the chosen model
    text_tokenizer = PegasusTokenizer.from_pretrained(model_identifier)

    # Load the Pegasus model for conditional generation (summarization)
    summarization_model = PegasusForConditionalGeneration.from_pretrained(model_identifier)

    # Tokenize the input text for the model, ensuring truncation and padding as needed
    tokenized_text = text_tokenizer(input_text, truncation=True, padding="longest", return_tensors="pt")

    # Generate a summary from the tokenized text using specified parameters
    generated_summary_ids = summarization_model.generate(
        **tokenized_text,
        max_length=max_summary_length,  # Maximum length of the generated summary
        min_length=min_summary_length,  # Minimum length of the generated summary
        length_penalty=penalty_for_length,  # Penalty to control summary length
        num_beams=beam_count  # Number of beams for beam search, improving summary quality
    )

    # Decode the generated token IDs back into a human-readable string without special tokens
    generated_summary_text = text_tokenizer.decode(generated_summary_ids[0], skip_special_tokens=True)

    return generated_summary_text  # Return the generated summary


# Function to generate a summary using the Transformers pipeline
def generate_summary_with_pipeline(input_text, model_identifier, min_summary_length=50, max_summary_length=150,
                                   penalty_for_length=2.0):
    # Load the Pegasus tokenizer specific to the chosen model
    text_tokenizer = PegasusTokenizer.from_pretrained(model_identifier)

    # Load the Pegasus model for conditional generation
    summarization_model = PegasusForConditionalGeneration.from_pretrained(model_identifier)

    # Initialize the summarization pipeline with the model and tokenizer
    summarization_pipeline = pipeline("summarization", model=summarization_model, tokenizer=text_tokenizer,
                                      framework="pt")

    # Use the pipeline to generate a summary with specified parameters
    pipeline_generated_summary = summarization_pipeline(
        input_text,
        min_length=min_summary_length,  # Minimum length for the summary
        max_length=max_summary_length,  # Maximum length for the summary
        length_penalty=penalty_for_length  # Penalty to control length of the summary
    )[0]["summary_text"]  # Extract the summary text from the pipeline output

    return pipeline_generated_summary  # Return the generated summary


# Sample text to summarize
sample_text_for_summary = (
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

# Model identifier for the Pegasus model used for summarization
model_identifier = "google/pegasus-xsum"

# Generate a summary directly using the model
direct_model_summary = generate_summary_from_model(sample_text_for_summary, model_identifier)
print("Direct Model Summary:", direct_model_summary)

# Generate a summary using the Transformers pipeline
pipeline_generated_summary = generate_summary_with_pipeline(sample_text_for_summary, model_identifier)
print("Pipeline Summary:", pipeline_generated_summary)
