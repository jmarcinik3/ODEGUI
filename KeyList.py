"""
This file contains the KeyList class.
This class stores keys for future access.
Keys are stored in dictionary.
Dictionary key is the prefix codename for each key.
Dictionary value is the collection of keys containing that prefix.
"""
from typing import List, Union

from Config import readPrefixes


class KeyList:
    """
    Stores list of keys for future access.
    Keys are generated as prefix + separator + tag.
    
    :ivar prefixes: dictionary from prefix codename to prefix
    :ivar separator: string separating prefix and tag
    :ivar keys: dictionary from prefix codename to all keys stored under that prefix codename
    """

    def __init__(self, separator: str = ' ') -> None:
        """
        Constructor for :class:`~KeyList.KeyList`
        
        :param separator: string separating prefix and tag
        """
        self.prefixes = readPrefixes()
        self.separator = separator
        self.keys = {}

    def getSeparator(self) -> str:
        """
        Get string that separates prefix and tag.
        
        :param self: :class:`~KeyList.KeyList` to retrieve separator from
        """
        return self.separator

    def getPrefix(self, prefix: str, with_separator: bool = False) -> str:
        """
        Get prefix from prefix codename.
        
        :param self: :class:`~KeyList.KeyList` to retrieve prefix from
        :param prefix: prefix codename to retrieve prefix of
        :param with_separator: set True to return prefix with separator.
            Set False to return prefix without separator.
        """
        prefix = self.prefixes[prefix]
        if with_separator:
            prefix += self.getSeparator()
        return prefix

    def getPrefixes(self):
        """
        Get all prefixes stored in key list.
        
        :param self: :class:`~KeyList.KeyList` to retrieve prefix codenames from
        """
        return self.keys.keys()

    def addKey(self, prefix: str, tag: str) -> None:
        """
        Add new key to key list.
        
        __Recursion Base__
            add single key: keys [str]
        
        :param self: :class:`~KeyList.KeyList` to retrieve add key to
        :param prefix: prefix codename for new key
        :param tag: suffix tag for new key
        """
        if prefix not in self.getPrefixes():
            self.keys[prefix] = [tag]
        else:
            self.keys[prefix].append(tag)

    def generateKey(self, prefix: str, tag: str = None) -> str:
        """
        Generate name of new key from prefix and tag
        
        :param self: :class:`~KeyList.KeyList` to generate key name from
        :param prefix: prefix codename for new key
        :param tag: suffix tag for new key
        """
        if isinstance(tag, str):
            return self.getPrefix(prefix) + self.getSeparator() + tag
        elif tag is None:
            return self.getPrefix(prefix)
        else:
            raise TypeError("tag must be str")

    def getKeyList(self, prefixes: Union[str, List[str]] = None) -> List[str]:
        """
        Get keys stored in key list.
        
        :param self: :class:`~KeyList.KeyList` to retrieve key list from
        :param prefixes: only retrieve keys with this prefix codename.
            Acts as a filter.
        """
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        elif prefixes is None:
            prefixes = self.keys.keys()

        keys = [self.generateKey(prefix, tag) for prefix in prefixes for tag in self.keys[prefix]]
        return keys

    def getKey(self, prefix: str, tag: str = None, add_key: bool = True) -> str:
        """
        Get name of key in key list.
        Adds key to key list if not already stored, optional.
        
        :param self: :class:`~KeyList.KeyList` to get key name from
        :param prefix: prefix codename for new key
        :param tag: suffix tag for new key
        :param add_key: set True to add new to collection of keys.
            Set False otherwise.
        """
        key = self.generateKey(prefix, tag)
        if add_key and key not in self.getKeyList():
            self.addKey(prefix, tag)
        return key
