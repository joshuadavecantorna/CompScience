import math
from collections import defaultdict

# Sample email dataset (Email text, Class label)
emails = [
    ("Win a lottery now", "spam"),
    ("Earn money quickly", "spam"),
    ("Lowest mortgage rates available", "spam"),
    ("Congratulations you have won a prize", "spam"),
    ("Meeting scheduled for tomorrow", "ham"),
    ("Let's catch up over coffee", "ham"),
    ("Are you coming to the party", "ham"),
    ("Important project update", "ham"),
]

# Function to preprocess data: Convert to lowercase and split into words
def preprocess_data(data):
    processed_data = []
    for text, label in data:
        words = text.lower().split()
        processed_data.append((words, label))
    return processed_data

# Compute Prior Probabilities P(Class)
def compute_prior_probabilities(data):
    class_counts = defaultdict(int)
    total_emails = len(data)
    
    for _, label in data:
        class_counts[label] += 1
    
    return {label: count / total_emails for label, count in class_counts.items()}

# Compute Likelihoods P(Word | Class)
def compute_likelihoods(data):
    word_counts_per_class = defaultdict(lambda: defaultdict(int))
    class_word_totals = defaultdict(int)

    for words, label in data:
        for word in words:
            word_counts_per_class[label][word] += 1
            class_word_totals[label] += 1

    # Apply Laplace Smoothing
    likelihoods = {}
    for label, word_counts in word_counts_per_class.items():
        total_words = class_word_totals[label]
        likelihoods[label] = {
            word: (count + 1) / (total_words + len(word_counts))  # Laplace smoothing
            for word, count in word_counts.items()
        }
    
    return likelihoods, class_word_totals

# Naïve Bayes Classifier for Email
def classify_email(email, prior, likelihood, class_word_totals):
    words = email.lower().split()
    class_probs = {}

    for label in prior:
        log_prob = math.log(prior[label])  # Start with log prior probability

        # Compute log likelihood for each word in the email
        for word in words:
            if word in likelihood[label]:
                log_prob += math.log(likelihood[label][word])
            else:
                # Handle unseen words with small probability
                log_prob += math.log(1 / (class_word_totals[label] + len(likelihood[label])))

        class_probs[label] = log_prob

    return max(class_probs, key=class_probs.get)  # Return class with highest probability

# Preprocess data
processed_data = preprocess_data(emails)

# Compute priors and likelihoods
prior_probabilities = compute_prior_probabilities(processed_data)
likelihoods, class_word_totals = compute_likelihoods(processed_data)

# Test email classification
test_emails = [
    "Win a free prize now",  # Expected: Spam
    "Lowest rates on loans available",  # Expected: Spam
    "Team meeting scheduled for Monday",  # Expected: Ham
    "Let's grab some coffee this weekend"  # Expected: Ham
]

for email in test_emails:
    predicted_class = classify_email(email, prior_probabilities, likelihoods, class_word_totals)
    print(f'The email "{email}" is classified as: {predicted_class}')