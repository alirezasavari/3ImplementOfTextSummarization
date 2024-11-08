import spacy
import pytextrank

# Load the spaCy language model with a large English vocabulary and word vectors
nlp_pipeline = spacy.load("en_core_web_lg")

# Add the TextRank algorithm to the spaCy pipeline for extracting key phrases and summarization
nlp_pipeline.add_pipe("textrank")

# The text input we want to process and summarize using TextRank
processed_text = nlp_pipeline(
    "The resulting edges of the graph are weighted. "
    "You then run a complex graph-based ranking formula over this weighted graph to determine "
    "the most important sentences in the original text and create the final summary. "
    "The math behind this formula is outside the scope of this article, but if you are curious, "
    "take a look at chapter 2.2."
)

# Display the summary, which contains the most relevant sentences from the text
print("Summary:")
# Use TextRank's summary method to extract the top-ranked sentences for summarization
# - `limit_phrases=3`: Limits key phrases considered for ranking
# - `limit_sentences=5`: Limits the number of summary sentences to display
for summary_sentence in processed_text._.textrank.summary(limit_phrases=3, limit_sentences=5):
    print(summary_sentence)

# Print a visual separator
print("\n" + "-" * 70)
print("Top ten phrases with their ranks:")

# Extract top-ranked phrases from the processed text, showing each phrase with its rank score
# This creates a list of tuples with (phrase, rank), showing how important each phrase is in context
top_ranked_phrases = [(phrase.chunks[0], phrase.rank) for phrase in processed_text._.phrases]
print(top_ranked_phrases[:10])  # Display only the top ten ranked phrases
