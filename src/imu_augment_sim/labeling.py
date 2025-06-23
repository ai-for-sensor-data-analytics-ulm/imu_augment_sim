"""Helpers for defining and evaluating automatic labeling rules."""

from typing import Any, Callable

class Rule:
    """Single logical rule used during automatic labelling.

    Parameters
    ----------
    name : str
        Human readable name of the rule.
    func : Callable[..., bool]
        Callable returning ``True`` or ``False`` when applied to ``data``.
    kwargs : dict[str, Any]
        Keyword arguments forwarded to ``func`` during evaluation.
    """

    def __init__(self, name: str, func: Callable[..., bool], kwargs: dict[str, Any]):
        self.name = name
        self.func = func
        self.kwargs = kwargs

    def evaluate(self, data) -> bool:
        """Evaluate the rule on the provided data.

        Parameters
        ----------
        data : Any
            Arbitrary data structure passed to ``func``.

        Returns
        -------
        bool
            Result of ``func``.
        """
        try:
            return self.func(data=data, **self.kwargs)
        except Exception as e:
            raise RuntimeError(f"Fehler in Regel '{self.name}': {e}")


class RuleEvaluator:
    """Evaluate a set of :class:`Rule` objects using a custom logic function.

    Parameters
    ----------
    data : Any
        Data passed to each rule during evaluation.
    rules : dict[str, Rule]
        Mapping of rule names to :class:`Rule` instances.
    logic : Callable[[dict[str, bool]], bool]
        Function that combines individual rule results into a final label.
    """

    def __init__(self, data, rules: dict[str, Rule], logic: Callable[[dict[str, bool]], bool]):
        self.data = data
        self.rules = rules
        self.logic = logic

    def evaluate(self) -> tuple[bool, dict[str, bool]]:
        """Evaluate all rules and aggregate the results.

        Returns
        -------
        tuple[bool, dict[str, bool]]
            Tuple containing the aggregated label and individual rule results.
        """
        results = {name: rule.evaluate(self.data) for name, rule in self.rules.items()}
        return self.logic(results), results









