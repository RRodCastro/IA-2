from fractions import Fraction
spam = 'spam'
ham = 'ham'
ignore_characters = ['.', ',', ';',
                     '\n', '!', '?', '/', ':', ')', '(']


class Analyzer():
    def __init__(self, lines):
        # dict with ocurrence of each word
        self.words_dict = {ham: {}, spam: {}}
        # dict with number of ham and spam messages
        self.count_type_ocurrence = {ham: 0, spam: 0}
        # dict with number of total words
        self.total_words_count = {ham: 0, spam: 0}
        self.parse_data(lines)

    def count_words(self, type, words):
        """
        Input: ham|spam, and an array of words
        words_dict: {ham: {word1: 10, word2: 10}, spam: {word3: 10, word4: 10}}
        """
        self.count_type_ocurrence[type] += 1
        for word in words:
            if (word in self.words_dict[type]):
                self.words_dict[type][word] += 1
            else:
                self.words_dict[type][word] = 1

    def clean_message(self, message):
        """
        Input: Dirty string message
        Output: Clean string message
        """
        for character in ignore_characters:
            message = str(message).replace(character, '')
        return message.lower().split(' ')

    def sum_words(self, message_type):
        """
        Input: message_type
        Output: total words on message_type
        """
        return (sum(self.words_dict[message_type][element]
                    for element in self.words_dict[message_type]))

    def parse_data(self, lines):
        """
        Input: Lines
        Out: Dictonaries with data
        """
        for line in lines:
            # SPAM || HAM
            message = line.split("\t")
            message_content = self.clean_message(message[1])
            self.count_words(message[0], message_content)
        self.total_words_count[ham] = self.sum_words(ham)
        self.total_words_count[spam] = self.sum_words(spam)
        self.total_messages = sum(self.count_type_ocurrence.values())
        # P(SPAM) and P(HAM)
        self.type_probability = {ham: Fraction(self.count_type_ocurrence[ham], self.total_messages), spam: Fraction(
            self.count_type_ocurrence[spam], self.total_messages)}

    def get_probabilty_from_word(self, message_type, word, k):
        """
        Given a word and the message type, the probability that the message belongs to the
        """
        if (word in self.words_dict[message_type]):
            return (
                Fraction(self.words_dict[message_type][word] + k,
                         self.total_words_count[message_type] + 2 * k)
            )
        return (Fraction(0 + k,
                         self.total_words_count[message_type] + 2 * k)
                )

    def get_probabilty_from_message(self, message_content, k):
        """
        Input: a message and k-factor
        Out: Probability that message is spam or ham (>0.50 is spam and <0.5 is ham)
        """
        message_content = self.clean_message(message_content)
        spam_probabilty = self.type_probability[spam]
        ham_probability = self.type_probability[ham]
        for word in message_content:
            spam_probabilty = spam_probabilty * \
                self.get_probabilty_from_word(spam, word, k)
            ham_probability = ham_probability * \
                self.get_probabilty_from_word(ham, word, k)
        return Fraction(spam_probabilty, spam_probabilty + ham_probability)
