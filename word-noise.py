def eliminate_vowels(word):
    vowels = 'aeiou'
    return ''.join(c for c in word if c not in vowels)

def remove_duplicate_consonants(word):
    cleaned_word = []
    for i, char in enumerate(word):
        if i == 0 or char != word[i - 1]:
            cleaned_word.append(char)
    return ''.join(cleaned_word)

def generate_noisy_form(word, rule):
    if rule == 1:
        noisy_word = eliminate_vowels(word)
    elif rule == 2:
        noisy_word = remove_duplicate_consonants(word)
    return noisy_word

input_word = input()
rule = int(input())
noisy_form = generate_noisy_form(input_word, rule)
print(noisy_form)
