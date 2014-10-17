import elephant.conditions as conditions
import types
from neo.core.container import unique_objs


class NeoFilter(object):
    """
    Wrapper class for :module: elephant.core.conditions

    Wraps conditions from :module: elephant.core.conditions to give an easy
    access and to enable execution of multiple conditions.

    See also
    --------
    elephant.core.conditions
    """

    def __init__(self, container, apply_func=None, **kwargs):
        """
        Parameters
        ----------
        container : neo Object
            An arbitrary neo Object to analyze.
        apply_func: function
            A user-defined filter-function which is applied to given input.
        """
        self._input = container
        # Set the conditions
        self._d_conditions = {}
        self.__set_default_conditions()
        # List of filtered object
        self._filt_obj = []
        if apply_func is not None:
            self.apply_function(apply_func)
        if kwargs is not None:
            self.set_conditions(**kwargs)

    def apply_function(self, func):
        # TODO allow a list of functions
        filt_res = func(self._input)
        if filt_res:
            self._filt_obj = self.__intersect_hash(filt_res,
                                                   self._filt_obj)
        # Conditions does not hold
        else:
            self._filt_obj = []

    def __set_default_conditions(self):
        # Get all module function names
        l = [conditions.__dict__.get(i) for i in dir(conditions)]
        for i in l:
            # Check if they are a (not built-in) function
            if isinstance(i, types.FunctionType):
                if not i.__name__.startswith(('__', '_')):
                    self._d_conditions[i.__name__] = (False, )
        self.__apply_conditions()

    def set_conditions(self, **kwargs):
        """
        The available conditions and their default parameters are (key/value):
            :key                    :var
            trial_has_n_st:         (False, 0)
            trial_has_n_as:         (False, 0)
            trial_has_exact_st:     (False, 0)
            trial_has_exact_as:     (False, 0)
            trial_has_n_rc:         (False, 0)
            trial_has_n_units:      (False, 0)
            trial_has_no_overlap:   (False, 0)
            each_st_has_n_spikes:   (False, 0)
            contains_each_unit:     (False, 0)
            contains_each_rc:       (False, 0)
            data_aligned:           (False, 0)
        """
        for key in kwargs.keys():
            # Check for a valid key
            if key not in self._d_conditions:
                raise ValueError("Keyword: %s not valid." % str(key))
        # Set conditions in dictionary
        for (cond, default) in self._d_conditions.items():
            self._d_conditions[cond] = kwargs.get(cond, default)
        self.__apply_conditions()

    def reset_conditions(self):
        """
        Resets the conditions to default state.

        Resets the values of the condition dictionary and the to the default
        state.

        See also:
        __set_default_conditions : The method reset_trial_condition calls the
        __set_default_conditions() method to reset above mentioned dictionaries
        to their default state.
        """
        self._d_conditions.clear()
        del self._filt_obj[:]
        self.__set_default_conditions()

    def __apply_conditions(self, _input=None):
        # For user defined filter method
        if not _input:
            _input = self._input
        # first_filter = True
        for key in self._d_conditions:
            if self._d_conditions[key][0]:
                # Get corresponding func from module
                func = getattr(conditions, key)
                # if first_filter:
                #     # Assume for a function with two parameter as input
                #     try:
                #         filt_res = func(_input,
                #                         **self._d_conditions[key][1])
                #         first_filter = False
                #     # If the function does not support more than one parameter
                #     except IndexError:
                #         filt_res = func(_input)
                # else:
                #     try:
                #         filt_res = func(self._filt_obj,
                #                         **self._d_conditions[key][1])
                #     except IndexError:
                #         filt_res = func(self._filt_obj)
                try:
                    filt_res = func(self._input,
                                    **self._d_conditions[key][1])
                except IndexError:
                    filt_res = func(self._input)
                # if filt_res not in self._filt_obj and filt_res:
                if filt_res:
                    self._filt_obj = self.__intersect_hash(filt_res,
                                                           self._filt_obj)
                    # self._filt_obj.extend(filt_res)
                # Conditions does not hold
                else:
                    self._filt_obj = []
                    return

    @property
    def conditions(self):
        """
        Returns a dictionary of set conditions.

        Returns
        -------
        conditions: dict
            Dictionary of set and default conditions.

        Notes
        -----
        The user defined function is not included in the dictionary as a
        condition.
        """
        return self._d_conditions

    @property
    def filtered(self):
        """
        Returns a list of filtered, unique neo objects.

        Returns
        -------
        filtered : list of neo.core objects
            Returns the filtered list of unique objects in a list.
        """
        # return unique_objs(
        #     [item for sublist in self._filt_obj for item in sublist])
        return unique_objs([item for item in self._filt_obj])

    # @staticmethod
    # TODO Remove or check if this method is faster, should not
    # def __intersect(inp, lst):
    #     result = []
    #     if lst:
    #         for i in lst:
    #             take = False
    #             tmp = None
    #             for j in inp:
    #                 if type(j) == type(i):
    #                     take = True
    #                     tmp = j
    #                     break
    #             if take:
    #                 result.append(i)
    #                 if i is not tmp:
    #                     result.append(tmp)
    #     else:
    #         result.extend(inp)
    #     return result

    @staticmethod
    def __intersect_hash(a, b):
        """
        Intersects two lists **type** based using a hash-table for look-ups.

        Parameters
        ----------
        a : list
            Actual filtered results list
        b: list
            List from prior filtered results

        Returns
        -------
        result : list
            A list which only includes intersected objects.
        """
        result = []
        if b:
            # Transform a to hash-table/dictionary
            ht = {}
            for i in a:
                if not type(i) in ht:
                    ht[type(i)] = type(i)
            for j in b:
                if type(j) in ht:
                    result.append(j)
        else:
            result.extend(a)
        return result