import sys

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import torch
import functools
import sympy as sp
import abc
from abc import abstractmethod
from call_function_with_timeout import SetTimeout
from sklearn.linear_model import Lasso

import SRConfig
from SRData import SRData

x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9")


def init_np_seed(seed):
    np.random.seed(seed)


class MyTorchVariable:
    def __init__(self, maxArity, weights, mask: np.array, name):
        self.v = torch.nn.Parameter(
            torch.tensor(weights, dtype=torch.float64), requires_grad=True
        )
        # self.mask = mask
        self.mask = torch.from_numpy(mask).to(dtype=torch.float64)
        # self.mask_orig = copy.deepcopy(mask)
        self.mask_orig = self.mask.clone()
        self.maxArity = maxArity
        self.name = name
        self.a: torch.tensor
        return

    def hasAnyActiveInput(self, inputLayerActiveNodes: list[bool]):
        """
        Checks if the variable has any non-zero pair of weight-mask
        :return: True, if the node has some active input. False, otherwise.
        """
        if isinstance(self.mask, np.ndarray):
            self.mask = torch.tensor(self.mask, dtype=torch.int)
        res = self.v * self.mask
        res = res * torch.tensor(inputLayerActiveNodes, dtype=torch.float64)
        return torch.any(res).item()

    def setRandomWeights(
        self,
        layerId: int,
        inputLayerActiveNodes: list[bool],
        perturbation: bool = False,
    ):
        """
        Generates an array of weights using only a random subset of inputLayerActiveNodes.
        Assigns the new values and sets the mask accordingly.
        :return: True, if the mask has changed. False, otherwise.
        """
        mask_before = self.mask.clone()
        origMask = torch.tensor(inputLayerActiveNodes, dtype=torch.float64).reshape(
            len(inputLayerActiveNodes), 1
        )
        origMaskNonZero = torch.nonzero(origMask, as_tuple=True)
        if len(origMaskNonZero[0]) == 0:
            self.set_allzero_value_mask()
            if not torch.equal(mask_before, self.mask):
                return True
            return False
        # ---
        tmp = SRConfig.r.choice(origMaskNonZero[0], 1)
        if len(tmp) == 0:
            self.set_allzero_value_mask()
            if not torch.equal(mask_before, self.mask):
                return True
            return False
        # ---
        nonzero = tmp[0]
        finalMask = torch.tensor(
            [int(SRConfig.r.random() < 1.0) if el == 1 else 0 for el in origMask]
        ).reshape(
            origMask.shape
        )  # --- ORIGINAL: 0.5. TODO: 1.0
        if torch.sum(finalMask) == 0:
            finalMask[nonzero] = 1

        if perturbation:
            values = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma / 10.0,
                size=self.v.shape,
                biasWeight=(" b" in self.name),
            )  # --- TODO: normal distribution respecting the minimum init vale
        else:
            values = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma,
                size=self.v.shape,
                biasWeight=(" b" in self.name),
            )  # --- TODO: normal distribution respecting the minimum init vale
        # ---
        values = values * finalMask
        self.set_value_mask(values)
        if not torch.equal(mask_before, self.mask):
            return True
        return False

    def randomize_originally_zero_weights(self):
        """
        For every position where mask_orig == 1 but v == 0 (i.e. the position
        is structurally permitted yet currently inactive), draw a new value
        uniformly from (-SRConfig.weight_sigma, SRConfig.weight_sigma) and
        write it into v.  mask is updated to reflect the new non-zero entries.

        Positions where mask_orig == 0 are never touched (structurally absent).
        Positions that already carry a non-zero value are left unchanged.
        """
        nv = self.v.detach().clone()
        # --- Positions allowed by the original mask that are currently zero
        permitted = self.mask_orig.bool()  # mask_orig == 1
        currently_zero = nv.abs() < SRConfig.activeWeightThreshold  # v ≈ 0
        targets = permitted & currently_zero

        n_targets = int(targets.sum().item())
        if n_targets == 0:
            return

        # --- Uniform samples in (-weight_sigma, weight_sigma)
        raw = SRConfig.r.uniform(
            low=-SRConfig.weight_sigma,
            high=SRConfig.weight_sigma,
            size=n_targets,
        )
        # --- Avoid accidental exact zeros (vanishingly unlikely, but safe)
        raw[raw == 0.0] = SRConfig.weight_sigma * 1e-6

        nv[targets] = torch.tensor(raw, dtype=nv.dtype)

        with torch.no_grad():
            self.v.copy_(nv)
        self.mask = (nv.abs() >= SRConfig.activeWeightThreshold).to(
            dtype=torch.float64, device=self.v.device
        )

    def adjustArity(self, mask):
        while torch.sum(mask != 0) > self.maxArity:
            nonzero_indices = torch.nonzero(mask, as_tuple=True)[0]
            to_zero = SRConfig.r.choice(nonzero_indices.tolist(), 1)[0]
            mask[to_zero] = 0
        return mask

    def set_allzero_value_mask(self):
        self.v = torch.zeros_like(self.v, requires_grad=True)
        self.mask = torch.zeros_like(self.mask)

    def set_value_mask(self, newValues):
        self.v = newValues.clone().detach().to(torch.float64).requires_grad_(True)
        self.mask = (torch.abs(newValues) >= SRConfig.activeWeightThreshold).to(
            torch.int
        )

    def adjust_value_mask_for_max_arity(self):
        """
        Prune current self.v so that at most self.maxArity values have abs(v) >= SRConfig.activeWeightThreshold.
        Excess active values are randomly zeroed (reproducible via SRConfig.r).
        Updates self.mask accordingly.
        """
        # --- Work on a detached copy of current weights (same device/dtype)
        nv = self.v.detach().clone()

        # --- Determine active values using threshold
        active_mask = nv.abs() >= SRConfig.activeWeightThreshold
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()
        n_active = len(active_indices)

        if n_active > self.maxArity:
            # --- Randomly choose which active indices to KEEP (reproducible)
            # keep_indices = SRConfig.r.sample(active_indices, self.maxArity)
            keep_indices = SRConfig.r.choice(
                active_indices, self.maxArity, replace=False
            ).tolist()
            # --- Build final keep mask
            keep_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            keep_mask[keep_indices] = True
            # --- Zero out the ones not kept
            nv[~keep_mask] = 0.0
            active_mask = keep_mask
        # --- Update parameter in-place (keeps optimizer references intact)
        with torch.no_grad():
            self.v.copy_(nv)
        # --- Update mask (float64, same device)
        self.mask = active_mask.to(dtype=torch.float64, device=self.v.device)

    def adjust_weights_to_prev_layer_active_nodes(
        self, prevLayerActiveNodes: list[bool]
    ):
        """
        Adjust self.v so that non-zero values appear only where prevLayerActiveNodes == True.

        Rules:
        1) Zero out all positions where prevLayerActiveNodes == False.
        2) Preserve existing non-zero values at allowed positions.
        3) Add new random non-zero values at allowed zero positions so that
        the final number of non-zero elements matches the original one.

        Use: w.adjust_wv_to_prev_layer_activity(prevLayerActiveNodes)

        Parameters
        ----------
        prevLayerActiveNodes : list[bool]
            Boolean activity mask for previous layer.
        """

        allowed = np.array(prevLayerActiveNodes, dtype=bool)
        # log(f"allowed ({allowed.shape}): {allowed.flatten()}", level="HIGHLIGHT_GREEN")
        curr_mask = self.mask.detach().cpu().numpy().copy().astype(bool)
        orig_nnz = int(curr_mask.sum())  # --- Original number of non-zero elements
        # log(f"orig_nnz: {orig_nnz}", level="HIGHLIGHT_GREEN")

        # --- 1) Zero out all disallowed positions
        v_np = self.v.detach().cpu().numpy().copy()  # --- Current weights as NumPy copy
        # log(f"v_np before: {v_np.flatten()}", level="HIGHLIGHT_GREEN")
        v_np[~allowed] = 0.0
        # v_np_1d = v_np.ravel()   # --- Not used any more when selecting candidate_idx
        # log(f"v_np_1d ({v_np_1d.shape}): {v_np_1d}", level="HIGHLIGHT_GREEN")
        curr_nnz = int(
            np.count_nonzero(v_np)
        )  # --- Count how many non-zero values remain
        # log(f"curr_nnz: {curr_nnz}", level="HIGHLIGHT_GREEN")

        # --- 2) Refill allowed zero positions so total nnz matches original nnz
        # log(f"prevLayerActiveNodes ({sum(prevLayerActiveNodes)}): {prevLayerActiveNodes}", level="HIGHLIGHT_GREEN")
        # log(f"orig_nnz= {orig_nnz}, curr_nnz= {curr_nnz}", level="HIGHLIGHT_GREEN")
        n_to_add = min(sum(allowed), orig_nnz - curr_nnz)
        if n_to_add > 0:
            # candidate_idx = np.where(allowed & (v_np_1d == 0.0))[0]
            candidate_idx = np.where(allowed)[0]
            # log(f"n_to_add= {n_to_add}, curr_nnz= {curr_nnz}", level="HIGHLIGHT_GREEN")
            # log(f"candidate_idx (shape={candidate_idx.shape}): {candidate_idx}", level="HIGHLIGHT_GREEN")
            chosen_idx = SRConfig.r.choice(candidate_idx, size=n_to_add, replace=False)
            # log(f"chosen_idx (shape={chosen_idx.shape}): {chosen_idx}", level="HIGHLIGHT_CYAN")
            # --- New values ~ N(0, 0.1)
            new_vals = SRConfig.r.normal(
                loc=0.0, scale=SRConfig.weight_sigma, size=n_to_add
            )
            # --- Avoid accidental exact zeros (extremely unlikely, but safe)
            new_vals[new_vals == 0.0] = 1e-12
            # log(f"new_vals (shape={new_vals.shape}): {new_vals}", level="HIGHLIGHT_CYAN")
            v_np[chosen_idx, 0] = new_vals

        # --- Write back into the existing Parameter
        with torch.no_grad():
            self.v.copy_(
                torch.as_tensor(v_np, dtype=self.v.dtype, device=self.v.device)
            )

        # --- Keep mask consistent with w.v
        new_mask = (v_np != 0.0).astype(np.float64)
        self.mask = torch.as_tensor(new_mask, dtype=torch.float64, device=self.v.device)

    def set_allzero_mask(self):
        self.mask = torch.zeros_like(self.mask)

    def set_mask(self, mask):
        self.mask = mask.clone()


class SymbolicExpression:

    def __init__(
        self,
        strExpr,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        self.strExpr = strExpr.replace("np.", "")
        self.symExpr = "0"
        self.simpleExpr = "0"
        func_with_timeout = SetTimeout(self.simplifyFormula, timeout=timeout)
        if self.strExpr != "":
            is_done: bool = False
            if simplify:
                is_done, is_timeout, erro_message, results = func_with_timeout(
                    self.strExpr
                )
            if is_done:
                self.simpleExpr = results
            else:
                self.simpleExpr = "NA"
        else:
            self.symExpr = "NA"
            self.simpleExpr = "NA"

    def simplifyFormula(self, formula):
        self.symExpr = sp.sympify(formula)
        simpleFormula = sp.simplify(self.symExpr)
        return simpleFormula


class NNUnit(abc.ABC):
    """
    Abstract class for NNUnits with different activation functions.
    """

    mu = 0.0

    @abstractmethod
    def __init__(self):
        self.name: str = ""
        self.inId: int = (
            -1
        )  # --- for UnitIdent1: the variable id (e.g., 0 for x0, 1 for x1, etc.). For other units: -1
        self.proxyName: str = ""
        self.isVariable: bool = (
            False  # --- True only for UnitIdent1 units representing variables
        )
        self.nbOfVars: int = 1
        self.learnable: bool = (
            True  # --- Default is that the unit's weight are learnable
        )
        self.weights: list[MyTorchVariable] = []  # --- list of MyTFVariables
        # self.maxArity: int = 3  # --- Maximum number of active input weights
        # self.maxArity: int = SRConfig.max_arities.get(self.name, 100)  # --- Maximum number of active input weights
        self.a: torch.Tensor = None  # --- node's output
        self.string_analytic_expression: str = ""
        self.symbolic_expression: SymbolicExpression = None

    @abstractmethod
    def define_weight_and_bias_vars(self, layerId: int, inSize: int):
        pass

    def reset_weights_and_bias(self):
        pass

    @abstractmethod
    def forward_pass(self, x_input):
        pass

    @abstractmethod
    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        """
        Generates string representation of the node using a single forward pass procedure.
        """
        pass

    @staticmethod
    def generate_normal_distribution(
        layerId: int, mu: float, sigma: float, size, biasWeight: bool = False
    ):
        if (
            (layerId == 0)
            and (SRConfig.r.random() < SRConfig.weight_init_one)
            and (not biasWeight)
        ):
            # -----------------------------------------------
            # --- +/- 1
            # -----------------------------------------------
            v = torch.tensor(SRConfig.r.normal(loc=mu, scale=sigma, size=size))
            # --- negative values
            neg_mask = v < 0.0
            v = torch.where(neg_mask, torch.tensor(-1.0), v)
            # --- positive values
            pos_mask = v > 0.0
            v = torch.where(pos_mask, torch.tensor(1.0), v)
        else:
            # -----------------------------------------------
            # --- Normal distribution random values
            # -----------------------------------------------
            v = SRConfig.r.normal(loc=mu, scale=sigma, size=size)
            v = torch.tensor(v)
            # --- negative values
            neg_mask = v < 0.0
            v = torch.where(neg_mask, -SRConfig.weight_min_init + v, v)
            # --- positive values
            pos_mask = v > 0.0
            v = torch.where(pos_mask, SRConfig.weight_min_init + v, v)
        return v

    @staticmethod
    def create_tf_variable(
        shape,
        name,
        layerId: int,
        maxArity=sys.maxsize,
        weights=None,
        mask=None,
        offset=0.0,
    ):
        if weights is None:
            # --- TODO: normal distribution respecting the minimum value
            weights = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma,
                size=shape,
                biasWeight=(" b" in name),
            )
        # ---
        if mask is None:
            mask = np.ones(shape=shape)
        weights = np.multiply(weights, mask)
        tf_v = MyTorchVariable(maxArity=maxArity, weights=weights, mask=mask, name=name)
        return tf_v


class UnitMultiply(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, layerId: int, inSize):
        """
        Called in NNLayer.add_unit()
        :param inSize:
        :return:
        """
        self.W1: MyTorchVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.W2: MyTorchVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.weights = [self.W1, self.W2]
        return [self.W1, self.W2], None

    def forward_pass(self, x_input: torch.Tensor):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W1.v):
                self.W2.v = self.W2.v.clone().zero_()
            elif not torch.any(self.W2.v):
                self.W1.v = self.W1.v.clone().zero_()

        # ---
        z1 = torch.matmul(x_input, self.W1.v)
        z2 = torch.matmul(x_input, self.W2.v)
        self.a = torch.mul(z1, z2)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} * {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitMultiply_b(NNUnit):
    nbOfVars = 4

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 4
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, layerId: int, inSize):
        """
        Called in NNLayer.add_unit()
        :param inSize:
        :return:
        """
        self.W1: MyTorchVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b1 = NNUnit.create_tf_variable(
            shape=(1, 1), name=self.name + " " + "b1", layerId=-1
        )
        self.W2: MyTorchVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.b2 = NNUnit.create_tf_variable(
            shape=(1, 1), name=self.name + " " + "b2", layerId=-1
        )
        self.weights = [self.W1, self.b1, self.W2, self.b2]
        return [self.W1, self.b1, self.W2, self.b2], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not (torch.any(self.W1.v) or torch.any(self.b1.v)):
                self.W2.v = self.W2.v.clone().zero_()
                self.b2.v = self.b2.v.clone().zero_()
            elif not (torch.any(self.W2.v) or torch.any(self.b2.v)):
                self.W1.v = self.W1.v.clone().zero_()
                self.b1.v = self.b1.v.clone().zero_()
        # ---
        z1 = torch.add(torch.matmul(x_input, self.W1.v), self.b1.v)
        z2 = torch.add(torch.matmul(x_input, self.W2.v), self.b2.v)
        self.a = torch.mul(z1, z2)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]

        if np.abs(self.b1.v.numpy(force=True).flatten()[0]) > SRConfig.activeWeightThreshold:
            op1.append("{:2.15}".format(self.b1.v.numpy(force=True).flatten()[0]))
        if np.abs(self.b2.v.numpy(force=True).flatten()[0]) > SRConfig.activeWeightThreshold:
            op2.append("{:2.15}".format(self.b2.v.numpy(force=True).flatten()[0]))
        # ---
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} * {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitDivide(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # --- 4
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # --- W and b for two inner potentials
        self.W1 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.W2 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.weights = [self.W1, self.W2]
        return [self.W1, self.W2], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W1.v):
                self.W2.v = self.W2.v.clone().zero_()
            elif not torch.any(self.W2.v):
                self.W1.v = self.W1.v.clone().zero_()
        # ---
        div_thresh = 1.0e-4
        z1 = torch.matmul(x_input, self.W1.v)
        z2 = torch.matmul(x_input, self.W2.v)
        mask = (z2 > div_thresh).to(torch.float64)
        div = 1.0 / (torch.abs(z2) + 1e-15)
        self.a = torch.mul(z1, div)
        self.a = torch.mul(self.a, mask)
        return self.a, z2

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} / {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitDivide_b(NNUnit):
    nbOfVars = 4

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 4
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # --- W and b for two inner potentials
        self.W1 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b1 = NNUnit.create_tf_variable(
            shape=(1, 1), name=self.name + " " + "b1", layerId=-1
        )
        self.W2 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.b2 = NNUnit.create_tf_variable(
            shape=(1, 1), name=self.name + " " + "b2", layerId=-1
        )
        self.weights = [self.W1, self.b1, self.W2, self.b2]
        return [self.W1, self.b1, self.W2, self.b2], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not (torch.any(self.W1.v) or torch.any(self.b1.v)):
                self.W2.v = self.W2.v.clone().zero_()
                self.b2.v = self.b2.v.clone().zero_()
            elif not (torch.any(self.W2.v) or torch.any(self.b2.v)):
                self.W1.v = self.W1.v.clone().zero_()
                self.b1.v = self.b1.v.clone().zero_()
        # ---
        div_thresh = 1.0e-4
        z1 = torch.add(torch.matmul(x_input, self.W1.v), self.b1.v)
        z2 = torch.add(torch.matmul(x_input, self.W2.v), self.b2.v)
        mask = (z2 > div_thresh).to(torch.float64)
        div = 1.0 / (torch.abs(z2) + 1e-15)
        self.a = torch.mul(z1, div)
        self.a = torch.mul(self.a, mask)
        return self.a, z2

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]

        if np.abs(self.b1.v.numpy(force=True).flatten()[0]) > SRConfig.activeWeightThreshold:
            op1.append("{:2.15}".format(self.b1.v.numpy(force=True).flatten()[0]))
        if np.abs(self.b2.v.numpy(force=True).flatten()[0]) > SRConfig.activeWeightThreshold:
            op2.append("{:2.15}".format(self.b2.v.numpy(force=True).flatten()[0]))
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} / {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitReciprocal(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # --- W and b for two inner potentials
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.W.v = self.W.v.clone().zero_()
        # ---
        div_thresh = 1.0e-4
        z = torch.matmul(x_input, self.W.v)
        mask = (torch.abs(z) > div_thresh).to(torch.float64)
        div = 1.0 / (torch.abs(z) + 1e-15)
        self.a = torch.mul(div, mask)
        return self.a, z

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            self.string_analytic_expression = f"(1 / {op1})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSin(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1  # ---
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # ---
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()
        # ---
        if torch.sum(self.W.mask) == 0.0:
            self.a = torch.zeros((x_input.shape[0], 1), dtype=torch.float64)
        else:
            z = torch.matmul(x_input, self.W.v)
            self.a = torch.sin(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            self.string_analytic_expression = f"np.sin({op1})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSin_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # ---
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # ---
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()
        # ---
        if torch.sum(self.W.mask) == 0.0:
            self.b.set_allzero_mask()
            self.a = torch.zeros((x_input.shape[0], 1), dtype=torch.float64)
        else:
            z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
            self.a = torch.sin(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if self.b.mask:
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        else:
            op1 = "0"
        # ---

        if op1:
            self.string_analytic_expression = f"np.sin({op1})"
        else:
            self.string_analytic_expression = ""
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitCos(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        if (torch.sum(self.W.mask) == 0.0) and (
            not torch.any(torch.abs(self.W.v) > 0.0)
        ):
            self.a = torch.zeros((x_input.shape[0], 1), dtype=torch.float64)
        else:
            z = torch.matmul(x_input, self.W.v)
            self.a = torch.cos(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]

        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        else:
            op1 = "0"
        # ---
        if op1:
            self.string_analytic_expression = f"np.cos({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitCos_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        if (torch.sum(self.W.mask) == 0.0) and (
            not torch.any(torch.abs(self.W.v) > 0.0)
        ):
            self.b.set_allzero_mask()
            self.a = torch.zeros((x_input.shape[0], 1), dtype=torch.float64)
        else:
            z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
            self.a = torch.cos(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if self.b.mask:
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        else:
            op1 = "0"
        # ---
        if op1:
            self.string_analytic_expression = f"np.cos({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitTanh(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        # ---
        z = torch.matmul(x_input, self.W.v)
        self.a = torch.tanh(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.tanh({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitTanh_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.b.v = self.b.v.clone().zero_()
        # ---
        z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        self.a = torch.tanh(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]

        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.tanh({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitArcTan(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        # ---
        z = torch.matmul(x_input, self.W.v)
        self.a = torch.atan(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.arctan({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitArcTan_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.b.v = self.b.v.clone().zero_()
        # ---
        z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        self.a = torch.atan(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.arctan({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSquare(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        # ---
        z = torch.matmul(x_input, self.W.v)
        self.a = torch.square(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**2"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSquare_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.b.v = self.b.v.clone().zero_()
        # ---
        z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        self.a = torch.square(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**2"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSqrt(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        # ---
        sqrt_thresh = 1.0e-6
        z = torch.matmul(x_input, self.W.v)
        mask = (z > sqrt_thresh).to(torch.float64)
        self.a = torch.sqrt(torch.abs(z))
        self.a = torch.mul(self.a, mask)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**0.5"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSqrt_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.b.v = self.b.v.clone().zero_()
        # ---
        sqrt_thresh = 1.0e-6
        z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        mask = (z > sqrt_thresh).to(torch.float64)
        self.a = torch.sqrt(torch.abs(z))
        self.a = torch.mul(self.a, mask)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**0.5"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitCube(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        # ---
        z = torch.matmul(x_input, self.W.v)
        self.a = torch.pow(z, 3.0)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**3"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitCube_b(NNUnit):
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        with torch.no_grad():
            if not torch.any(self.W.v):
                self.b.v = self.b.v.clone().zero_()
        # ---
        z = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        self.a = torch.pow(z, 3.0)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**3"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitSign(NNUnit):
    nbOfVars = 1

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, layerId: int, inSize: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()
        # ---
        if torch.sum(self.W.mask) == 0.0:
            self.a = torch.zeros((x_input.shape[0], 1), dtype=torch.float64)
        else:
            z = torch.matmul(x_input, self.W.v)
            self.a = torch.tanh(
                100 * z
            )  # --- approximation of the sign function by tanh
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            self.string_analytic_expression = f"np.sign({op1})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitIdent(NNUnit):  # UnitIdent_b
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        self.a = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitIdent1(NNUnit):  # UnitIdent
    nbOfVars = 1

    def __init__(self, inId=0):
        super().__init__()
        self.inId = inId
        self.learnable = False
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = 1  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # --- weights
        w = np.zeros(shape=(inSize, 1))
        w[self.inId] = 1.0
        # --- mask
        m = np.zeros(shape=(inSize, 1))
        m[self.inId] = 1.0
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            weights=w,
            mask=m,
            layerId=layerId,
        )
        self.weights = [self.W]
        return None, [self.W]  # --- W and b are not being learned

    def reset_weights_and_bias(self):
        # --- weights
        w = torch.zeros_like(self.W.v)
        w[self.inId] = 1.0
        self.W.set_value_mask(w)

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        self.a = torch.matmul(x_input, self.W.v)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{}".format(i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"{op1}"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression


class UnitIdentLS(NNUnit):  # UnitIdent_b
    nbOfVars = 2

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = SRConfig.max_arities.get(
            self.name, sys.maxsize
        )  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float64)
        x_input = x_input.double()

        self.a = torch.add(torch.matmul(x_input, self.W.v), self.b.v)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self,
        currLayer,
        layers,
        simplify: bool = False,
        timeout: int = SRConfig.simplificationTimeout,
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy(force=True).flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if (
            np.abs(self.b.v.numpy(force=True).flatten()[0])
            > SRConfig.activeWeightThreshold
        ):
            op1.append(
                "{:2.15}".format(
                    round(
                        self.b.v.numpy(force=True).flatten()[0],
                        SRConfig.precisionDigits,
                    )
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify, timeout=timeout
        )
        return self.string_analytic_expression

    def apply_least_squares(self, x_input):
        y_data = SRData.y_data

        model = Lasso(alpha=SRConfig.lassoAlpha, fit_intercept=True, max_iter=10000)
        model.fit(x_input, y_data)
        W = torch.tensor(model.coef_, dtype=torch.float64).view(-1, 1)
        b = torch.tensor(model.intercept_, dtype=torch.float64).view(1, 1)

        self.W.set_value_mask(W)
        self.b.set_value_mask(b)
        return


# ================= EXTRAS =============================================================================================
