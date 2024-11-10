from transformers import PegasusForConditionalGeneration , PegasusTokenizer , BartForConditionalGeneration , \
    BartTokenizer , pipeline


# Function to generate a summary using a pre-trained model
def generate_summary_from_model(input_text , model_identifier , model_class , tokenizer_class , max_summary_length=250 ,
                                min_summary_length=150 ,
                                penalty_for_length=2.0 , beam_count=4) :
    # Load the selected model's tokenizer and model
    text_tokenizer = tokenizer_class.from_pretrained ( model_identifier )
    summarization_model = model_class.from_pretrained ( model_identifier )

    # Tokenize the input text for the model, ensuring truncation and padding as needed
    tokenized_text = text_tokenizer ( input_text , truncation=True , padding="longest" , return_tensors="pt" )

    # Generate a summary from the tokenized text using specified parameters
    generated_summary_ids = summarization_model.generate (
        input_ids=tokenized_text["input_ids"] ,
        attention_mask=tokenized_text["attention_mask"] ,
        max_length=max_summary_length ,
        min_length=min_summary_length ,
        length_penalty=penalty_for_length ,
        num_beams=beam_count
    )

    # Decode the generated token IDs back into a human-readable string without special tokens
    generated_summary_text = text_tokenizer.decode ( generated_summary_ids[0] , skip_special_tokens=True )

    return generated_summary_text


# Function to generate a summary using the Transformers pipeline
def generate_summary_with_pipeline(input_text , model_identifier , tokenizer_class , model_class ,
                                   min_summary_length=50 , max_summary_length=150 ,
                                   penalty_for_length=2.0) :
    # Load the selected model's tokenizer and model
    text_tokenizer = tokenizer_class.from_pretrained ( model_identifier )
    summarization_model = model_class.from_pretrained ( model_identifier )

    # Initialize the summarization pipeline with the model and tokenizer
    summarization_pipeline = pipeline ( "summarization" , model=summarization_model , tokenizer=text_tokenizer ,
                                        framework="pt" )

    # Use the pipeline to generate a summary with specified parameters
    pipeline_generated_summary = summarization_pipeline (
        input_text ,
        min_length=min_summary_length ,
        max_length=max_summary_length ,
        length_penalty=penalty_for_length
    )[0]["summary_text"]

    return pipeline_generated_summary


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

# Model configurations for each summarization model
models = [
    ("google/pegasus-xsum" , PegasusForConditionalGeneration , PegasusTokenizer , "Pegasus") ,
    ("facebook/bart-large-cnn" , BartForConditionalGeneration , BartTokenizer , "BART") ,
    ("sshleifer/distilbart-cnn-12-6" , BartForConditionalGeneration , BartTokenizer , "DistilBART")
]

# Execute and display summaries for each model
for model_identifier , model_class , tokenizer_class , model_name in models :
    print ( f"--- {model_name} Model Summaries ---\n" )

    # Generate a summary directly using the model
    direct_model_summary = generate_summary_from_model ( sample_text_for_summary , model_identifier , model_class ,
                                                         tokenizer_class )
    print ( f"Direct Model Summary ({model_name}):\n{direct_model_summary}\n" )

    # Separator between the two summaries of each model
    print ( "-------\n" )

    # Generate a summary using the Transformers pipeline
    pipeline_generated_summary = generate_summary_with_pipeline ( sample_text_for_summary , model_identifier ,
                                                                  tokenizer_class , model_class )
    print ( f"Pipeline Summary ({model_name}):\n{pipeline_generated_summary}\n" )

    # Separator between models
    print ( "=======" * 10 + "\n" )
