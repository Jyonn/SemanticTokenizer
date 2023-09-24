from UniTok import UniDep
from oba import Obj


class ItemDepot:
    def __init__(
            self,
            depot: str,
            order,
            append=None,
    ):
        self.depot = UniDep(depot)
        self.order = Obj.raw(order)
        self.append = Obj.raw(append) or []
