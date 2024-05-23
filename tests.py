from spelling_confusion_matrices import error_tables
from spell_checker import Spell_Checker, normalize_text
import os
import random

########################################################################
#                               Tests                                  #
########################################################################

# Test text normalization
def test_text_normalization():
    examples = [
        "He's not going to the zoo tomorrow.",
        "I've been working 9-5 at a 5.5% increase since 2020-01-01",
        "Dr. Smith prescribed 50mg of Ibuprofen, twice a day.",
        "Can't we just 'normalize' this text? Please, please, please!",
        "It’s a wonderful life, isn’t it? There’s nothing like a good book.",
        "Copyright laws are changing all over the world."
    ]

    for i, example in enumerate(examples):
        normalized = normalize_text(example)
        print(f"Example {i+1}: {normalized}")

def test_text_generation(file_path, contexts):
    spell_checker = Spell_Checker()  
    lm = Spell_Checker.Language_Model(n=3, chars=False)
    text = load_text_file(file_path)
    lm.build_model(text)

    # Build the model with your loaded text
    spell_checker.add_language_model(lm)
    for context in contexts:
        n = random.randint(5, 25)
        generated_text = spell_checker.lm.generate(context, n)
        print(f"\nGenerated Text of size {n}:\n", generated_text)
        print(f"text size is {len(generated_text.split())}")
    

# Load text file utility function
def load_text_file(file_path):
    """Read the entire contents of a text file into a single string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Test language model and spell checker
def test_spell_checker(file_path, error_table):
    spell_checker = Spell_Checker()  
    lm = Spell_Checker.Language_Model(n=3, chars=False)
    text = load_text_file(file_path)

    # Build the model with your loaded text
    lm.build_model(text)
    print('LM model initiated based on the provided text file')

    # Generate text (LM test)
    sample_text = "do not remove it"  # Test case to generate text
    generated_text = lm.generate(sample_text, n=50)
    print("Generated Text:", generated_text)
    
    # Build spell checker
    spell_checker.add_language_model(lm)
    spell_checker.add_error_tables(error_table)
    
    # Test log-likelihood evaluation
    text = "This is a test."
    print(normalize_text(text))
    log_likelihood = spell_checker.evaluate_text(text)
    print(f'Log-likelihood of "{text}" is: {log_likelihood} based on LM')
    log_likelihood = spell_checker.evaluate_text(text.split()[0])
    print(f'Log-likelihood of "{text.split()[0]}" is: {log_likelihood} based on LM')
    
    # Test correction probability
    word = "his"
    alpha = 0.95
    candidate = "him"
    probability = spell_checker._P(candidate, word, '', alpha)
    print(f"\nThe probability of correcting '{word}' to '{candidate}' based on LM is {probability}")
    probability = spell_checker._P(candidate, word, 'never mind', alpha)
    print(f"\nWhen context is added, the probability is {probability}")
    
    # Test spell_check function
    test_cases = [
        "spe1l",
        "spe11",
        "do not remove iit",
        "do not remve it",
        "do no remove it",
        "helo from the other side",
        "hello from the oter side",
        'keepig with the news of victory which were convyed',
        "spel check",
        "the uqick brown fox",
        "the quick brn fox",
        "This is an xampel",
        "It was a bright and suni day.",
        "Definately going to the party.",
        "smiled with a sense of his superiority over a weec woman"
    ]

    alpha = 0.95

    for text in test_cases:
        corrected_text = spell_checker.spell_check(text, alpha, len(text.split()))
        print(f'\nOriginal: "{text}"\nCorrected: "{corrected_text}"')

def main():
    # Set file paths and load error table
    file_path = 'Corpora/Norvigs_big_text_file.txt'
    # file_path = 'Corpora/trump_historical_tweets.txt'
    
    if not os.path.exists(file_path):
        print(f"Error: The file path '{file_path}' does not exist.")
        return

    # Run tests
    print("\nRunning text generation test:")
    test_text_generation(file_path, [None, 
                                     'Should I stay?',
                                     'The men and officers returning spoke of a brilliant victory',
                                     'The quick brown fox stopped at the bar for a sniff',
                                     "Yo Ho, Yo Ho, a pirate is"])
    
    print("\nRunning text normalization tests:")
    test_text_normalization()

    print("\nRunning spell checker tests:")
    test_spell_checker(file_path, error_tables)

if __name__ == "__main__":
    main()
