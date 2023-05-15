"""
6.1010 Spring '23 Lab 9: Autocomplete
"""

# NO ADDITIONAL IMPORTS!
import doctest
from text_tokenize import tokenize_sentences


class PrefixTree:
    """
    Recursive data structure that maps keys
    to values in a way that allows for
    efficient searching
    """

    def __init__(self):
        self.value = None
        self.children = {}

    def __setitem__(self, key, value, add=False):
        """
        Add a key with the given value to the prefix tree,
        or reassign the associated value if it is already present.
        Raise a TypeError if the given key is not a string.

        If the add parameter is True then the value is appended
        to the current value; if the value is None, then the value
        is set to the passed-in value as a default
        >>> t = PrefixTree()
        >>> t['bla'] = 1
        >>> t['bl'] = 2
        >>> print(t)
        "None", (b : "None", (l : "2", (a : "1")))
        >>> t['bla'] = 4
        >>> print(t)
        "None", (b : "None", (l : "2", (a : "4")))
        """
        if not isinstance(key, str):
            raise TypeError("key is not of type string")
        last_node, k_index = self.find_item_helper(key)
        # if complete key is not found, adds key;
        # else, skips loop and reassigns key that was found
        for i in range(k_index, len(key)):
            # create a series of nodes to
            # fill-in the rest of the key
            new_node = PrefixTree()
            last_node.children[key[i]] = new_node
            last_node = new_node
        if add and last_node.value is not None:
            last_node.value += value
        else:
            last_node.value = value

    def __getitem__(self, key):  # 4:40
        """
        Return the value for the specified prefix.
        Raise a KeyError if the given key is not in the prefix tree.
        Raise a TypeError if the given key is not a string.
        >>> t = PrefixTree()
        >>> t['bla'] = 1
        >>> t['blo'] = 2
        >>> t['bel'] = 3
        >>> print(t['blo'])
        2

        """
        if not isinstance(key, str):
            raise TypeError("key is not a string")
        last_node, k_index = self.find_item_helper(key)
        if k_index == len(key):
            return last_node.value
        else:
            raise KeyError("key not in tree")

    def __delitem__(self, key):
        """
        Delete the given key from the prefix tree if it exists.
        Raise a KeyError if the given key is not in the prefix tree.
        Raise a TypeError if the given key is not a string.

        As of now, the tree structure remains as it was, just
        without the value stored at the key location
        >>> t = PrefixTree()
        >>> t['bla'] = 1
        >>> t['blo'] = 2
        >>> print(t)
        "None", (b : "None", (l : "None", (a : "1"), (o : "2")))
        >>> del t['bla']
        >>> print(t)
        "None", (b : "None", (l : "None", (a : "None"), (o : "2")))
        """
        if not isinstance(key, str):
            raise TypeError("key is not of type string")
        node, k_index = self.find_item_helper(key)
        if k_index == len(key) and node.value is not None:
            node.value = None
        else:
            raise KeyError("key not in tree")

    def __contains__(self, key):
        """
        Is key a key in the prefix tree?  Return True or False.
        Raise a TypeError if the given key is not a string.
        >>> t = PrefixTree()
        >>> t['bla'] = 1
        >>> t['blo'] = 2
        >>> t['bel'] = 3
        >>> print('bla' in t)
        True
        >>> print('ble' in t)
        False
        >>> print('bl' in t)
        False
        """
        if not isinstance(key, str):
            raise TypeError("key is not of type string")
        node, k_index = self.find_item_helper(key)
        if k_index == len(key) and node.value is not None:
            return True
        return False

    def __iter__(self, key=""):
        """
        Generator of (key, value) pairs for all keys/values in this prefix tree
        and its children.  Must be a generator!
        >>> t = PrefixTree()
        >>> t['bat'] = 7
        >>> t['bar'] = 3
        >>> t['bark'] = ':)'
        >>> t[''] = 2
        >>> print(set(t) == {('bat', 7), ('bar', 3), ('bark', ':)'), ('', 2)})
        True
        """
        # could be value in first node
        if self.value is not None:
            yield (key, self.value)
        for c_key in self.children:
            yield from self.children[c_key].__iter__(key + c_key)

    def find_item_helper(self, key):
        """
        Returns the furthest PrefixTree node
        accessible down the key path, and the
        index that marks the extent of the
        key which led there
        """
        # PrefixTree -> data and children
        current = self
        k_index = 0
        while k_index < len(key) and key[k_index] in current.children:
            current = current.children[key[k_index]]
            k_index += 1
        return current, k_index

    def __str__(self):
        """
        Returns a string format for the PrefixTree
        """
        tree = f'"{self.value}"'
        for prefix, child in self.children.items():
            tree += f", ({prefix} : {child.__str__()})"
        return tree


def word_frequencies(text):
    """
    Given a piece of text as a single string, create a prefix tree whose keys
    are the words in the text, and whose values are the number of times the
    associated word appears in the text.
    >>> story = "one own, one one"
    >>> tree = word_frequencies(story)
    >>> print(tree)
    "None", (o : "None", (n : "None", (e : "3")), (w : "None", (n : "1")))
    """
    sentences = tokenize_sentences(text)
    sentences = [sent.split() for sent in sentences]

    tree = PrefixTree()
    for sent in sentences:
        for word in sent:
            # set tree at prefix to value
            # = tree.__setitem__()
            tree.__setitem__(word, 1, add=True)
    return tree


# should work
def autocomplete(tree, prefix, max_count=None):
    """
    Return the list of the most-frequently occurring elements that start with
    the given prefix.  Include only the top max_count elements if max_count is
    specified, otherwise return all.

    Raise a TypeError if the given prefix is not a string.

    >>> story = "one own bone, one one bone"
    >>> tree = word_frequencies(story)
    >>> print(set(autocomplete(tree, 'o', max_count=3)) == {'one', 'own'})
    True
    >>> print(set(autocomplete(tree, 'o', max_count=2)) == {'one', 'own'})
    True
    >>> print(set(autocomplete(tree, 'o')) == {'one', 'own'})
    True
    >>> print(set(autocomplete(tree, 'o', max_count=1)) == {'one'})
    True
    >>> print(set(autocomplete(tree, 'b')) == {'bone'})
    True
    """
    # the following code examines all words in a
    # subtree representing all children tied to a prefix,
    # sorts the list in decreasing order based on value
    # (frequency), and takes the top max_count number

    if not isinstance(prefix, str):
        raise TypeError("key is not of type string")
    entries = []
    # find genesis node -> node that matches key
    gen_node, k_index = tree.find_item_helper(prefix)
    # if prefix was found, otherwise return empty list
    if k_index == len(prefix):
        # appends prefix to all values found from prefix onward
        entries = [(prefix + key, value) for key, value in gen_node]
        # sorts in descending order
        entries.sort(key=lambda x: x[1], reverse=True)
        # takes out value information
        entries = [key for key, value in entries]
        # returns at most the number requested
        if max_count is not None:  # else, return them all
            return entries[:max_count]
    return entries


def autocorrect(tree, prefix, max_count=None):
    """
    Return the list of the most-frequent words that start with prefix or that
    are valid words that differ from prefix by a small edit.  Include up to
    max_count elements from the autocompletion.  If autocompletion produces
    fewer than max_count elements, include the most-frequently-occurring valid
    edits of the given word as well, up to max_count total elements.
    >>> story = "one own bno bone, o', one one bone bond"
    >>> tree = word_frequencies(story)
    >>> print(set(autocorrect(tree, 'one', max_count=3)) == {'one', 'bone'})
    True
    >>> print(set(autocorrect(tree, 'on', max_count=3)) == {'one', 'own', 'o'})
    True
    >>> print(set(autocorrect(tree, 'bon')) == {'bone', 'bno', 'bond'})
    True
    >>> print(set(autocorrect(tree, 'bon', max_count=2)) == {'bone', 'bond'})
    True
    >>> print(set(autocorrect(tree, 'bon', max_count=1)) == {'bone'})
    True
    """

    # why duplicate with str1 -> replace with same letter
    def create_edits(str1):
        """
        Returns a set containing all
        possible edits that can be made to str1
        """
        edits = set()
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for i in range(len(str1)+1):
            if i < len(str1):
                edits.add(str1[:i] + str1[i + 1 :]) # deletion
            if i < len(str1) - 1:
                edits.add(str1[:i] + str1[i + 1] + str1[i] + str1[i + 2 :]) # transpose
            for char in alphabet:
                edits.add(str1[:i] + char + str1[i:]) # insertion
                if i < len(str1):
                    consider = str1[:i] + char + str1[i + 1 :]
                    if consider != str1:
                        edits.add(consider) # replacement
        return edits

    auto_completed = set(autocomplete(tree, prefix, max_count))
    present_edits = []
    all_possible_edits = create_edits(prefix)

    # if you needto consider one, you need to
    # consider them all to find the most frequent
    if max_count is None or max_count - len(auto_completed) > 0:
        for edit in all_possible_edits:
            if edit in tree:
                present_edits.append((edit, tree[edit]))
 
    if max_count is not None and max_count - len(auto_completed) != 0:  # else, will append all of the edits found
        # remove duplicates from auto_completed
        for element in present_edits:
            if element[0] in auto_completed:
                present_edits.remove(element)
        # top frequency edits
        present_edits = sorted(present_edits, key=lambda x: x[1], reverse=True)
        present_edits = present_edits[:max_count - len(auto_completed)]

    return_list = auto_completed | {key for key, _ in present_edits}
    return list(return_list)


def word_filter(tree, pattern):
    """
    Return list of (word, freq) for all words in the given prefix tree that
    match pattern.  pattern is a string, interpreted as explained below:
         * matches any sequence of zero or more characters,
         ? matches any single character,
         otherwise char in pattern char must equal char in word.
    >>> story = "n main man in"
    >>> tree = word_frequencies(story)
    >>> print(set(word_filter(tree, '*n')) == {('main', 1), ('man', 1), ('in', 1), ('n', 1)})
    True
    >>> print(set(word_filter(tree, 'm**n')) == {('main', 1), ('man', 1)})
    True
    """
    # each time, look at the next char in the pattern:
    # - if letter, must match
    # - if ?, can be any single match
    # - if *, can be any single match, no match
    # (zero character), or skips to next
    # char to account for a match > 1
    # explore recursive subtrees of children;
    # two paths for 'b*a' case
    words = set()
    def recursive_helper(sub_tree, remaining_pattern, pre_key=""):
        # check for ability to reduce strings of stars
        if len(remaining_pattern) > 1 and remaining_pattern[0] == '*':
            while len(remaining_pattern) > 1 and remaining_pattern[1] == '*':
                remaining_pattern = remaining_pattern[1:]
        # starting below, looks to add word at the end of the
        # pattern, or when there is only one star left
        if len(remaining_pattern) == 0 or len(remaining_pattern) == 1 and remaining_pattern[0] == "*":
            if sub_tree.value is not None:
                words.add((pre_key, sub_tree.value))
        if len(remaining_pattern) > 0:  # *n -> 'n' w/ child of 
            for key, child in sub_tree.children.items():
                if remaining_pattern[0] == "*":
                    # accepts another character under '*'
                    # -> different branches of recursion here explore
                    # all children, as if '*' is accounting for
                    # varying amounts of characters
    # add to guest list
                    # option 1 for 'b*a' situation
                    recursive_helper(child, remaining_pattern, pre_key + key)

                    # if there is a character after '*',
                    # there is an option to look at the
                    # next character -> ie. in this subtree
                    # '*' accounts for no characters,
                    # and the pattern is adjusted; ie
                    # same tree is passed with new pattern

                    # option 2 for 'b*a' situation
                    if len(remaining_pattern) > 1:
                        recursive_helper(sub_tree, remaining_pattern[1:], pre_key)
                else:
                    # remaining_pattern[0] == '?' or match with other character
                    if remaining_pattern[0] == "?" or remaining_pattern[0] == key:
                        recursive_helper(child, remaining_pattern[1:], pre_key + key)
                    # else -> required a specific
                    # character, but did not find
                    # it, so this subtree is not explored
    recursive_helper(tree, pattern)
    return list(words)


# you can include test cases of your own in the block below.
if __name__ == "__main__":
    doctest.testmod()
    with open("dracula.txt", encoding='utf-8') as f:
        text = f.read()
        tree = word_frequencies(text)

        total_words = 0
        for w in tree:
            total_words += w[1]
        print(total_words)
        # words = word_filter(tree, 'r?c*t')
        # print(words)
        # words.sort(key=lambda x:x[1], reverse=True)
        # words = [w for w, freq in words]
        # print(words[:6])
