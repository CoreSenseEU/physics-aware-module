import logging
import time
import warnings
import torch
import torch.nn as nn
import math
import numpy as np
import copy

import SRConfig
from SRData import SRData
from SRConstraints import SRConstraints
from NNLayer import NNLayer
from SRUnits import MyTorchVariable, NNUnit, UnitIdent1
from logger import align_text, log
from utils import NNTopology

warnings.filterwarnings("ignore")


class bsfSubTopology:
    def __init__(self, iter, losses, weights):
        self.iter = iter
        self.losses = losses
        self.weights = weights


class L05Regularizer(nn.Module):

    def __init__(self, strength, a):
        super(L05Regularizer, self).__init__()
        self.strength = strength
        self.a = a

    def __call__(self, x):
        x = torch.cat(x, 0)
        # --- |x| >= a
        a = torch.full_like(x, self.a)
        x1 = x[torch.abs(x) >= a]
        r1 = torch.sum(torch.sqrt(torch.abs(x1)))
        # --- |x| < a
        x2 = x[torch.abs(x) < a]
        c3 = torch.full_like(x2, 3.0)
        c4 = torch.full_like(x2, 4.0)
        c8 = torch.full_like(x2, 8.0)
        a1 = torch.full_like(x2, self.a)
        a3 = torch.pow(torch.full_like(x2, self.a), c3)
        t1 = -torch.pow(x2, c4) / (c8 * a3)
        t2 = (c3 * torch.square(x2)) / (c4 * a1)
        t3 = (c3 * a1) / c8
        r2 = torch.sum(torch.sqrt(t1 + t2 + t3))
        # ---
        s = torch.tensor(self.strength)
        res = torch.sum(s * (r1 + r2))
        return res


def getMutualDominance(
    first: dict[str, float], second: dict[str, float], criteria: list[str]
):
    """
    Check mutual dominance of the two subTopologies w.r.t. the given criteria set.
    :return:
    """
    firstIsBetter = False
    secondIsBetter = False
    firstDominates = False
    secondDominates = False
    firstEqualsSecond = False
    if len(criteria) == 1:
        if first[criteria[0]] < second[criteria[0]]:
            firstDominates = True
        elif first[criteria[0]] > second[criteria[0]]:
            secondDominates = True
        else:
            firstEqualsSecond = True
    else:
        for key in criteria:
            if first[key] < second[key]:
                firstIsBetter = True
            if first[key] > second[key]:
                secondIsBetter = True
        if firstIsBetter and (not secondIsBetter):
            firstDominates = True
        elif secondIsBetter and (not firstIsBetter):
            secondDominates = True
        elif (not firstIsBetter) and (not secondIsBetter):
            firstDominates = True
            firstEqualsSecond = True
    # ---
    return firstDominates, secondDominates, firstEqualsSecond


class SubTopology:
    keysPerfCplx: list[str] = ["valid_loss", "rmse_constr", "complexity"]
    keysAll: list[str] = ["valid_loss", "rmse_constr", "nbOfActiveNodes"]
    keysPerformance: list[str] = ["valid_loss", "rmse_constr"]

    def __init__(self):
        self.order = 0
        self.masterTopologyId: int = -1
        self.nnLayers: list[NNLayer] = []
        self.activeNodeCoordinates: list[list[bool]] = (
            []
        )  # --- list of node coordinates [layer, row] that will undergo the weights update
        self.nbOfActiveNodes: int = 0
        self.nbOfActiveWeights: int = 0
        self.nbOfActiveNodeOutputs: int = (
            0  # --- number of active nodes' output links (i.e., links that are on some path from some input to the output)
        )
        self.learnableParams: list[MyTorchVariable] = (
            []
        )  # --- parameters optimized by SGD
        self.regularizableParams: list = []
        self.params_2_learn: list[
            torch.nn.Parameter
        ]  # --- a list of tf.Variables that are tuned
        self.div_vars: list = []
        self.forwardFullMask = None
        self.front = SRConfig.maxFrontNb
        self.y_hat: torch.Tensor = None
        # --- topology complexity in terms of the number of active weights
        self.complexity: int = 0
        # --- Training RMSE
        self.last_loss_terms: list[float] = []
        self.last_loss_terms_mean = 0.0
        # --- Regularization
        self.last_regularization_terms = [
            torch.tensor(0.0) for _ in range(SRConfig.stack_size)
        ]
        self.reg_weight = SRConfig.reg2rmse
        # --- Constraints
        self.constraint_weight = SRConfig.constr2rmse
        self.constraint_terms: dict[str, float] = {}
        self.raw_constraint_terms: dict[str, float] = {}
        self.last_raw_constraints: dict[str, list[torch.Tensor]] = {}
        self.last_constraints_terms: dict[str, list[torch.Tensor]] = {}
        self.losses: dict[str, float] = {
            "valid_loss": -1.0,
            "train_loss": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0,
            "complexity": 0,
        }
        for constr in SRConstraints.constraintModules:
            self.last_raw_constraints[constr.name] = [
                torch.tensor(1.0) for _ in range(SRConfig.stack_size)
            ]
            self.last_constraints_terms[constr.name] = [
                torch.tensor(1.0) for _ in range(SRConfig.stack_size)
            ]
        self.nonDominatedImprovement = False

        self.performanceAcrossIterations: list[dict[str, float]] = (
            []
        )  # TODO: remove, just for testing purposes

    def __eq__(self, other):
        for l, layer in enumerate(self.nnLayers):
            for u, unit in enumerate(layer.units):
                for w, weight in enumerate(unit.weights):
                    if not torch.equal(
                        self.nnLayers[l].units[u].weights[w].v,
                        other.nnLayers[l].units[u].weights[w].v,
                    ):
                        return False
        return True

    def __hash__(self):
        summary = []
        for layer in self.nnLayers:
            for unit in layer.units:
                for weight in unit.weights:
                    t = weight.v.detach().cpu()
                    summary.append(float(torch.nansum(t)))  # Ignores NaNs
        return hash(tuple(summary))

    def cloneSubTopology(self):
        """
        Clones the subTopology.
        """
        subTopology = SubTopology()
        subTopology.activeNodeCoordinates = copy.deepcopy(self.activeNodeCoordinates)
        subTopology.masterTopologyId = self.masterTopologyId

        for i in range(len(self.nnLayers)):
            newLayer = self.nnLayers[i].clone_layer(
                layerId=i, activeNodes=self.activeNodeCoordinates
            )
            subTopology.learnableParams.extend(newLayer.learnableParams)
            subTopology.nnLayers.append(newLayer)

        subTopology.losses = {
            "valid_loss": -1.0,
            "train_loss": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0,
            "complexity": 0,
        }
        subTopology.nbOfActiveNodes = self.nbOfActiveNodes
        subTopology.nbOfActiveWeights = self.nbOfActiveWeights
        subTopology.complexity = self.complexity
        subTopology.front = copy.deepcopy(self.front)
        subTopology.div_vars = [var.clone() for var in self.div_vars]
        subTopology.y_hat = None
        # ---
        return subTopology

    def create_layer(self, layerId, in_size, unit_types=None, identities=True):
        nbOfLearnableWeights = 0
        nnLayer = NNLayer(in_size, layerId)
        # --- Units
        for u in unit_types.keys():
            for i in range(unit_types[u]):
                nbOfLearnableWeights += nnLayer.add_unit(u())
        # --- Identities
        if identities:
            for i in range(in_size):
                newUnit = UnitIdent1(i)
                if (layerId == 0) and (i < SRData.x_data.shape[1]):
                    newUnit.isVariable = True
                elif (layerId > 0) and (
                    self.nnLayers[layerId - 1].units[i].isVariable
                ):  # --- current UnitIdent1 points to a previous layer UnitIdent1 that represents a variable
                    newUnit.isVariable = True
                nnLayer.add_unit(newUnit)
        self.nnLayers.append(nnLayer)
        # ---
        return nnLayer

    @staticmethod
    def createSubtopologyFromParameters(
        activeNodesCoordinates, nnWeights, losses=None, text: str = ""
    ):
        """
        Creates a subTopology from the given 'nnDefinition' parameters.
        :param nnDefinition: self.activeNodeCoordinates, and a nested list of subTopology parameters 'nnParams'
        :return: SubTopology object
        """
        subTopology = SubTopology()
        subTopology.masterTopologyId = 0
        subTopology.activeNodeCoordinates = activeNodesCoordinates

        layer_defs = NNTopology.all_layer_defs[0]
        identities = NNTopology.all_identities[0]
        inputSize = SRData.x_data.shape[1]
        previousLayer = None
        for l, layerParams in enumerate(nnWeights):
            newLayer = NNLayer.create_layer_with_params(
                layerId=l,
                layerParams=layerParams,
                inputSize=inputSize,
                unit_types=layer_defs[l],
                identities=identities[l],
                previousLayer=previousLayer,
            )
            inputSize = len(newLayer.units)
            subTopology.learnableParams.extend(newLayer.learnableParams)
            subTopology.nnLayers.append(newLayer)
            previousLayer = newLayer
        subTopology.set_forwardpass_masks()

        # --- TODO: delete the lines below; they are just for debugging purposes.
        # reconstructed_validation_loss = subTopology.calculateValidationLoss()
        # if reconstructed_validation_loss != losses["valid_loss"]:
        #     log(
        #         f"!!! createSubtopologyFromParameters(), called from {text}: Reconstructed validation loss is incorrect, {losses['valid_loss']} --> {reconstructed_validation_loss}",
        #         start_time=SRConfig.start_time,
        #     )
        # --- END of debugging lines

        if losses is not None:
            subTopology.losses["valid_loss"] = losses["valid_loss"]
            subTopology.losses["rmse_constr"] = losses["rmse_constr"]
            subTopology.losses["nbOfActiveNodes"] = losses["nbOfActiveNodes"]
            subTopology.losses["complexity"] = losses["complexity"]
        return subTopology

    def getSubtopologyParameters(self, filePath=None):
        """
        Creates 'nnParams', which is a nested list. At the first level, it contains lists of layerParams, where each layerParams is a list of tuples in the form (unit.name, unitParams). For each unit, multiple tuples with the same unit.name are included—one for each variable associated with that unit.
        :return: tuple (self.masterTopologyId, self.activeNodeCoordinates, nnParams)
        """

        nnParams = []
        for l, layer in enumerate(self.nnLayers):
            layerParams = []
            for u, unit in enumerate(layer.units):
                for w in unit.weights:
                    layerParams.append((unit.name, w.v))
            nnParams.append(layerParams)

        # --- TODO: delete the lines below; they are just for debugging purposes.
        # reconstructed_subtopology = SubTopology.createSubtopologyFromParameters(
        #     self.activeNodeCoordinates, nnParams, self.losses
        # )
        # reconstructed_validation_loss = (
        #     reconstructed_subtopology.calculateValidationLoss()
        # )
        # if reconstructed_validation_loss != self.losses["valid_loss"]:
        #     log(
        #         f"!!! getSubtopologyParameters(): Reconstructed validation loss is incorrect, {self.losses['valid_loss']} --> {reconstructed_validation_loss}",
        #         start_time=SRConfig.start_time,
        #     )
        # --- END of debugging lines

        if filePath is not None:
            dump = {
                "activeNodesCoordinates": copy.deepcopy(self.activeNodeCoordinates),
                "nnParams": nnParams,
                "losses": copy.deepcopy(self.losses),
            }
            with open(filePath, "w") as f:
                f.write(f"{dump}\n")
            return dump

        return {
            "activeNodesCoordinates": copy.deepcopy(self.activeNodeCoordinates),
            "nnParams": nnParams,
            "losses": copy.deepcopy(self.losses),
        }

    def resetSubTopology(self):
        """
        Resets the whole subTopology so that all units are inactive.
        """
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                for w in node.weights:
                    w.set_allzero_value_mask()
                self.activeNodeCoordinates[l][n] = False
        self.nbOfActiveNodes = 0
        self.nbOfActiveWeights = 0
        self.nbOfActiveNodeOutputs = 0
        self.complexity = 0
        self.losses = {
            "valid_loss": -1.0,
            "train_loss": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0,
            "complexity": 0,
        }

    def setProxyNames(self):
        for l, layer in enumerate(self.nnLayers):
            # --- skip the output layer
            if l == len(self.nnLayers) - 1:
                return
            # ---
            for n, node in enumerate(layer.units):
                if "UnitIdent1" in self.nnLayers[l].units[n].name:
                    proxyNode: UnitIdent1 = self.nnLayers[l].units[n]
                    proxyId = proxyNode.inId
                    if l == 0:
                        proxyNode.proxyName = f"x[{proxyId}]"
                    elif "UnitIdent1" not in self.nnLayers[l - 1].units[proxyId].name:
                        proxyNode.proxyName = (
                            self.nnLayers[l - 1].units[proxyId].name
                        )  # --- node[l][n] points to an ordinary function unit
                    else:
                        proxyNode.proxyName = (
                            self.nnLayers[l - 1].units[proxyId].proxyName
                        )  # --- node[l][n] points to the UnitIdent1

    def resetFront(self):
        self.front = SRConfig.maxFrontNb

    def separate_learnable_nonlearnable_params(self):
        self.learnableParams = []
        self.regularizableParams = []
        for l, layer in enumerate(self.nnLayers):
            for u, unit in enumerate(layer.units):
                if (not self.activeNodeCoordinates[l][u]) or (not unit.learnable):
                    continue
                for w in unit.weights:
                    if torch.sum(w.mask) > 0.0:
                        self.learnableParams.append(w)
                        if " W" in w.name:
                            self.regularizableParams.append(w)

    def get_params_2_learn(self):
        self.params_2_learn = [p.v for p in self.learnableParams]
        self.params_2_regularize = [p.v for p in self.regularizableParams]
        self.params_2_regularize_shapes = self.get_shapes(self.params_2_regularize)

    def get_all_params(self):
        params = []
        for layer in self.nnLayers:
            for unit in layer.units:
                if unit.learnable:
                    for weight in unit.weights:
                        if weight.v.requires_grad:
                            params.append(weight.v)
        return params

    def set_forwardpass_masks(self):
        self.forwardFullMask = np.zeros(shape=(SRData.forwardpass_data.shape[0], 1))
        for k in SRData.forwardpass_data_boundaries.keys():
            m = copy.deepcopy(self.forwardFullMask)
            m[
                SRData.forwardpass_data_boundaries[k][
                    0
                ] : SRData.forwardpass_data_boundaries[k][1]
                + 1
            ] = 1.0
            SRData.forwardpass_masks[k] = m

    def layer_feed_forward(self, layer, x_input):
        a, z = layer.forward_pass(x_input)
        if z:
            self.div_vars.extend(z)
        return a, z

    def model_feed_forward(self, x_input):
        for i, l in enumerate(self.nnLayers):
            x_input, _ = self.layer_feed_forward(l, x_input)
        return x_input  # --- final layer output

    def applyLeastSquaresAndUpdate(self, logs, indId, start_time):
        # TODO aplikovat len ked to nezhorsi ziadnu z metrik a pocet aktivnych uzlov neklesne pod threshhold (?)
        logs.append(
            align_text(
                f"Least squares applied to individual {indId}.\n"
                f"Before LS valid loss: {self.losses['valid_loss']}, RMSE constr: {self.losses['rmse_constr']}, nbOfActiveNodes: {self.losses['nbOfActiveNodes']}",
                start_time=start_time,
            )
        )
        # --- least squares
        x_input = SRData.x_data
        for i, layer in enumerate(self.nnLayers):
            if i == len(self.nnLayers) - 1:
                break
            x_input, _ = self.layer_feed_forward(layer=layer, x_input=x_input)

        lastNode = self.nnLayers[-1].units[0]
        lastNode.apply_least_squares(x_input.detach().numpy())

        # --- clip weights and update losses
        self.updateLearnableParamsMaskAndValues()  # clip weights
        self.updateActiveNodes()
        self.losses["valid_loss"] = self.calculateValidationLoss()
        c_raw, c_weighted = self.calculateConstrainViolationsRMSE()
        self.losses["rmse_constr"] = c_raw

        logs.append(
            align_text(
                f"After LS valid loss: {self.losses['valid_loss']}, RMSE constr: {self.losses['rmse_constr']}, nbOfActiveNodes: {self.losses['nbOfActiveNodes']}",
                start_time=start_time,
            )
        )
        return

    def get_RMSE(self, y_hat, y_output, counts):
        if isinstance(y_output, torch.Tensor):
            y_output_tensor = y_output
        else:
            y_output_tensor = torch.tensor(y_output, dtype=torch.float64)
        loss = SRConfig.rmseScaleCoeff * torch.sqrt(
            torch.sum((y_output_tensor - y_hat) ** 2) / counts
        )
        return loss

    def get_MSE(self, y_hat, y_output, counts):
        if isinstance(y_output, torch.Tensor):
            y_output_tensor = y_output
        else:
            y_output_tensor = torch.tensor(y_output, dtype=torch.float64)
        loss = SRConfig.rmseScaleCoeff * (
            torch.sum((y_output_tensor - y_hat) ** 2) / counts
        )
        return loss

    def getAnalyticFormula(
        self, simplify: bool = False, timeout: int = SRConfig.simplificationTimeout
    ):
        """
        Constructs analytic expressions of all units.
        :return: Final analytic expression of the whole SubTopology.
        """
        for i, l in enumerate(self.nnLayers):
            for u in l.units:
                u.get_analytic_expression_forwardpass(
                    currLayer=i,
                    layers=self.nnLayers,
                    simplify=simplify,
                    timeout=timeout,
                )
        return self.nnLayers[-1].units[0].string_analytic_expression

    def getFormula(self):
        """
        Just returns the analytic formula represented by the SubTopology.
        :return: The analytic expression of the output node.
        """
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        return outputNode.string_analytic_expression

    def printAnalyticFormula(self, printSimplified=False):
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        print(f"\tAnalytic formula: {outputNode.string_analytic_expression}")
        if printSimplified:
            print(
                f"\tSimplified analytic formula: {outputNode.symbolic_expression.simpleExpr}"
            )

    def evaluateAnalyticFormula(self, x0, x1):
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        if len(outputNode.string_analytic_expression) > 0:
            return eval(outputNode.string_analytic_expression)
        else:
            return np.zeros(shape=(x0.shape))

    def get_shapes(self, reg_vars):
        shapes = [(v.shape[0], v.shape[1]) for v in reg_vars]
        list_set = set(shapes)
        shapes = list(list_set)
        return shapes

    def get_regularization_term(self, reg_vars=None, reg_lambda=0.1, shapes=None):
        if reg_lambda == 0.0:
            return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        # ---
        if shapes is None:
            shapes = self.get_shapes(reg_vars)
        # ---
        regularizer_L05 = L05Regularizer(strength=reg_lambda, a=0.01)
        # ---
        reg_terms = []
        for s in shapes:
            new_vars = [v for v in reg_vars if v.shape == s]
            reg_term = regularizer_L05(new_vars)
            reg_terms.append(reg_term)
        if reg_terms:
            return torch.sum(torch.stack(reg_terms))
        else:
            return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def calculate_constraints_terms_weight_new(self):
        sum = torch.tensor(0.0)
        for constr in SRConstraints.constraintModules:
            terms = torch.stack(self.last_constraints_terms[constr.name])
            mean = torch.nanmean(terms)
            sum += mean
        if sum == 0.0:
            self.constraint_weight = torch.tensor(1.0)
        else:
            self.constraint_weight = SRConfig.constr2rmse * (
                self.last_loss_terms_mean / sum
            )

    def add_constraint_terms(self, c_terms, raw_constraint_terms, updateWeights=False):
        sum = 0.0  # --- sum of normalized constraint terms
        for constr in SRConstraints.constraintModules:
            new_constraint_term = constr.module.get_constraint_term(
                constr=constr, data=SRData, y_hat=self.y_hat, weight=1.0
            )
            raw_constraint_terms[constr.name] = new_constraint_term.clone()
            self.last_raw_constraints[constr.name].append(
                new_constraint_term.clone().detach()
            )
            self.last_raw_constraints[constr.name] = self.last_raw_constraints[constr.name][1:]
            if SRConfig.adaptive:
                # --- 'm' is a mean of the last RAW constraint term values
                m = torch.mean(torch.stack(self.last_raw_constraints[constr.name]))
                if m > 0.0:
                    new_constraint_term /= m
            else:
                new_constraint_term = SRConfig.constr2rmse * new_constraint_term
            self.last_constraints_terms[constr.name].append(new_constraint_term.clone().detach())
            self.last_constraints_terms[constr.name] = self.last_constraints_terms[constr.name][1:]
            sum += new_constraint_term
            # ---
            c_terms[constr.name] = new_constraint_term.clone()

        if SRConfig.adaptive:
            # -----------------------------------
            # ------- Update constraint_weight
            # -----------------------------------
            if updateWeights:
                self.calculate_constraints_terms_weight_new()

            # ---------------------------------
            # ------- Weight c_terms and sum
            # ---------------------------------
            sum *= self.constraint_weight
            for constr in SRConstraints.constraintModules:
                c_terms[constr.name] *= self.constraint_weight

            # ---------------------------------
            # ------- Adjust c_terms
            # ---------------------------------
            if sum > SRConfig.constr2rmse * self.last_loss_terms_mean:
                r = sum / (SRConfig.constr2rmse * self.last_loss_terms_mean)
                for constr in SRConstraints.constraintModules:
                    c_terms[constr.name] = c_terms[constr.name] / r

        return c_terms, raw_constraint_terms

    def updateLearnableParamsMaskAndValues(self):
        """
        Updates p.mask and p.v based on current p.v
        """
        # --- create curr_params_mask based on actual p.v values
        curr_params_mask = [
            (p.v.abs() >= SRConfig.activeWeightThreshold).float()
            for p in self.learnableParams
        ]

        for m, p in zip(curr_params_mask, self.learnableParams):
            p.set_mask(np.multiply(m, p.mask))

        for p in self.learnableParams:
            with torch.no_grad():
                p.v *= p.mask

    def calculateValidationLoss(self):
        """
        Calculates defined loss on 'valid' data
        :return:
        """
        x_input = SRData.forwardpass_data
        x_input[
            SRData.forwardpass_data_boundaries["valid"][
                0
            ] : SRData.forwardpass_data_boundaries["valid"][1]
            + 1,
            :,
        ] = SRData.x_val

        self.y_hat = self.model_feed_forward(x_input)

        y_output = copy.copy(self.forwardFullMask)
        y_output[
            SRData.forwardpass_data_boundaries["valid"][
                0
            ] : SRData.forwardpass_data_boundaries["valid"][1]
            + 1,
            :,
        ] = SRData.y_val
        mask_tensor = torch.from_numpy(SRData.forwardpass_masks["valid"]).to(
            self.y_hat.device
        )
        valid = getattr(self, SRConfig.computeTrainValidLoss)(
            self.y_hat * mask_tensor, y_output, SRData.forwardpass_counts["valid"]
        )
        res = round(valid.item(), SRConfig.precisionDigits)
        return res

    def calculateConstrainViolationsRMSE(self, step: int = 1):
        self.constraint_terms = {}
        self.raw_constraint_terms = {}
        self.constraint_terms, self.raw_constraint_terms = self.add_constraint_terms(
            self.constraint_terms,
            self.raw_constraint_terms,
            updateWeights=(step % SRConfig.update_step == 0),
        )
        c_raw = float(np.sum([el.item() for el in self.raw_constraint_terms.values()]))
        c_weighted = float(np.sum([el.item() for el in self.constraint_terms.values()]))
        return round(c_raw, SRConfig.precisionDigits), round(
            c_weighted, SRConfig.precisionDigits
        )

    def checkUnitIdent1Nodes(self):
        """
        Checks if all UnitIdent1 nodes have at most one active input weight.
        :return:
        """
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if (self.activeNodeCoordinates[l][n]) and ("UnitIdent1" in node.name):
                    for w in node.weights:
                        if " b" not in w.name:
                            nonzero_count = torch.count_nonzero(w.v)
                            if nonzero_count > 1:
                                return False  # --- more than one active weights
        return True  # --- OK

    def findUnitsWithBelowThresholdOutput(self):
        """
        Finds all active units whose absolute output is below a data-relative
        threshold for every input vector, i.e. units that contribute negligibly
        to the (scaled) target.

        The threshold is data-driven: pruningOutputRelTol * std(y_train). It
        scales with the target's spread, so it is invariant to rescaling the
        targets (unlike the previous hard-wired absolute value). If std(y_train)
        is 0 (constant target) the threshold is 0, so only exactly-dead units
        are flagged.
        :return: list of tuples where each tuple is (l, n), i.e., coordinates of the below-threshold output unit
        """
        threshold = SRConfig.pruningOutputRelTol * SRData.target_scale_std()
        result = []
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:  # --- it is an active unit
                    aboveThresholdOutputValues = (
                        torch.abs(node.a) > threshold
                    ).int()
                    if sum(aboveThresholdOutputValues) == 0:
                        result.append((l, n))
        return result

    def pruneUnitsWithBelowThresholdOutput(self):
        """
        Prunes units that have below-threshold output
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        belowThresholdOutputNodes = self.findUnitsWithBelowThresholdOutput()
        # --- Inactivate belowThresholdOutputNodes
        if belowThresholdOutputNodes:
            for el in belowThresholdOutputNodes:
                self.activeNodeCoordinates[el[0]][el[1]] = False
                self.nbOfActiveNodes -= 1
                for w in self.nnLayers[el[0]].units[el[1]].weights:
                    w.set_allzero_value_mask()
            return True  # --- at least one un it has been pruned
        return False  # --- no pruning carried out

    def pruneWeights(self):
        """
        Prunes weights of each active unit ...
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        # -----------------------------------------------
        # --- Forward pass for all layers
        # -----------------------------------------------
        result = False  # --- flag if any weight was pruned
        for l, layer in enumerate(self.nnLayers):
            # --- create input to the l-th layer, i.e., output matrix composed of the 'l-1' layer's nodes output
            if l == 0:
                inputX = SRData.forwardpass_data
            else:
                inputX = np.empty(
                    (SRData.forwardpass_data.shape[0], len(self.nnLayers[l - 1].units))
                )
                for u, unit in enumerate(self.nnLayers[l - 1].units):
                    inputX[:, u] = unit.a.detach().numpy().reshape(-1)
            inputX = torch.tensor(inputX, dtype=torch.float64)
            # -----------------------------------------------
            # --- Find weights to be pruned in this layer
            # -----------------------------------------------
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    currResult = False
                    for w in node.weights:
                        if " b" not in w.name:
                            nbOfActiveWeights_old = int(torch.sum(w.mask).item())
                            inputTimesW = torch.abs(
                                inputX * w.v.T
                            )  # --- calculate values passed via all input edges to the variable 'w'
                            # --------------------------------------------------------------------------------------
                            # --- Variant A: Works with the largest absolute value of inputTimesW for each weight
                            # --------------------------------------------------------------------------------------
                            maxAbsValuePerWeight = torch.max(inputTimesW, dim=0).values
                            maxAbsValuePerWeight = maxAbsValuePerWeight.reshape(
                                (maxAbsValuePerWeight.shape[0], 1)
                            )
                            maxAbsWeight = torch.max(maxAbsValuePerWeight)
                            minAcceptableWeightAbsValue = (
                                SRConfig.pruningMinWeightRatio * maxAbsWeight
                            )  # --- minimum acceptable absolute value for the weight to stay active
                            # ---
                            okWeights = (
                                maxAbsValuePerWeight >= minAcceptableWeightAbsValue
                            ).to(torch.float64)
                            newWeights_A = w.v * okWeights
                            nbOfActiveWeights_new_A = torch.count_nonzero(
                                newWeights_A
                            ).item()
                            # ----------------------------------------------------------------------------
                            # --- Choose the variant
                            # ----------------------------------------------------------------------------
                            w.set_value_mask(newValues=newWeights_A)

                            if nbOfActiveWeights_old > nbOfActiveWeights_new_A:
                                currResult = True
                            if nbOfActiveWeights_new_A == 0:
                                self.activeNodeCoordinates[l][n] = False
                                self.nbOfActiveNodes -= 1
                    # if currResult:
                    #     self.forwardPass(data=SRData.forwardpass_data)  # --- Something changed in the subTopology, so update the self.y_hat
                    result = result or currResult
        return result

    def pruneWeights_new(self):
        """
        Prunes weights of each active unit.
        Weights with below-threshold value of abs(input_i * w_i) are set to zero and the variable's mask is updated accordingly.
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        # -----------------------------------------------
        # --- Forward pass over all layers
        # -----------------------------------------------
        result = False  # --- flag if any weight was pruned
        for l, layer in enumerate(self.nnLayers):
            # ---------------------------------------------------------------------
            # --- create input to the l-th layer,
            # --- i.e., output matrix composed of the 'l-1' layer's nodes output
            # ---------------------------------------------------------------------
            if l == 0:
                inputX = SRData.forwardpass_data
            else:
                inputX = np.empty(
                    (SRData.forwardpass_data.shape[0], len(self.nnLayers[l - 1].units))
                )
                for u, unit in enumerate(self.nnLayers[l - 1].units):
                    inputX[:, u] = inputX[:, u] = unit.a.detach().numpy().reshape(-1)
            inputX = torch.tensor(inputX, dtype=torch.float64)
            # -----------------------------------------------
            # --- Find weights to be pruned in this layer
            # -----------------------------------------------
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    currResult = False
                    # --- Node's dominant input-signal magnitude, used as the
                    #     relative basis for BOTH weight and bias pruning. Updated
                    #     in the weight branch below; the bias branch reads it.
                    node_signal_scale = 0.0
                    for w in node.weights:
                        if " b" not in w.name:
                            # ----------------
                            # --- Weights W
                            # ----------------
                            nbOfActiveWeights_old = int(torch.sum(w.mask).item())
                            inputTimesW = torch.abs(
                                inputX * w.v.T
                            )  # --- calculate values passed via all input edges to the variable 'w'
                            # --------------------------------------------------------------------------------------
                            # --- Works with the largest absolute value of inputTimesW for each weight
                            # --------------------------------------------------------------------------------------
                            maxAbsValuePerWeight = torch.max(inputTimesW, dim=0).values
                            maxAbsValuePerWeight = maxAbsValuePerWeight.reshape(
                                (maxAbsValuePerWeight.shape[0], 1)
                            )
                            maxAbsWeight = torch.max(maxAbsValuePerWeight)
                            node_signal_scale = max(
                                node_signal_scale, float(maxAbsWeight)
                            )
                            # --- Set the minimum acceptable value for the weight to stay active
                            # minAcceptableWeightAbsValue = SRConfig.pruningMinWeightRatio   # --- Absolute
                            minAcceptableWeightAbsValue = (
                                SRConfig.pruningMinWeightRatio * maxAbsWeight
                            )  # --- Relative
                            # ---
                            okWeights = (
                                maxAbsValuePerWeight >= minAcceptableWeightAbsValue
                            ).to(torch.float64)
                            newWeights = w.v * okWeights
                            nbOfActiveWeights_new = torch.count_nonzero(
                                newWeights
                            ).item()
                            # ----------------------------------------------------------------------------
                            # --- Choose the variant
                            # ----------------------------------------------------------------------------
                            w.set_value_mask(newValues=newWeights)
                            if nbOfActiveWeights_old > nbOfActiveWeights_new:
                                currResult = True
                            if nbOfActiveWeights_new == 0:
                                self.activeNodeCoordinates[l][n] = False
                                self.nbOfActiveNodes -= 1
                        else:
                            # ----------------
                            # --- Bias b
                            # ----------------
                            # Prune the bias only when its magnitude is negligible
                            # relative to the node's dominant input signal - the
                            # same relative basis used for the weights above. Uses
                            # abs() so a (legitimately) negative bias is not pruned
                            # solely for its sign.
                            if abs(w.v.detach().item()) < (
                                SRConfig.pruningMinWeightRatio * node_signal_scale
                            ):
                                w.set_allzero_value_mask()
                                currResult = True
                    result = result or currResult
        return result

    def forwardPass(self, data):
        x_input = data
        self.y_hat = self.model_feed_forward(x_input)

    @staticmethod
    def piecewise_constant_lr(epoch, lr_boundaries, lr_values):
        for i, boundary in enumerate(lr_boundaries):
            if epoch < boundary:
                return lr_values[i] / lr_values[0]
        return lr_values[-1] / lr_values[0]

    def get_subtopology_output(self, data):
        x_input = torch.tensor(data, dtype=torch.float64)
        y_hat = self.model_feed_forward(x_input)
        return y_hat

    def check_training_invariants(self, params_orig, logs):
        """
        Checks if the subtopology parameters, active node coordinates, and losses remain unchanged
        during training with learningSteps==0.
        """
        params_cur = self.getSubtopologyParameters()
        for layer1, layer2 in zip(
            params_orig["activeNodesCoordinates"], params_cur["activeNodesCoordinates"]
        ):
            if not np.array_equal(
                np.asarray(layer1, dtype=bool), np.asarray(layer2, dtype=bool)
            ):
                logs.append(
                    align_text(
                        f"!!! activeNodesCoordinates changed during training with learningSteps==0.\n"
                        f"Original: {params_orig['activeNodesCoordinates']}\n"
                        f"Current: {params_cur['activeNodesCoordinates']}",
                        start_time=SRConfig.start_time,
                    )
                )
                break

        for layer1, layer2 in zip(params_orig["nnParams"], params_cur["nnParams"]):
            for param1, param2 in zip(layer1, layer2):
                if not torch.allclose(param1[1], param2[1], rtol=1e-9):  # weights
                    logs.append(
                        align_text(
                            "!!! nnParams changed during training with learningSteps==0.",
                            start_time=SRConfig.start_time,
                        )
                    )

        for key in params_orig["losses"].keys():
            loss1 = params_orig["losses"][key]
            loss2 = params_cur["losses"][key]
            if loss1 in {0, -1} or loss2 in {0, -1}:
                continue

            if not math.isclose(loss1, loss2, rel_tol=1e-9):
                logs.append(
                    align_text(
                        f"!!! {key} changed during training with learningSteps==0."
                        f"\nOriginal: {loss1}\n"
                        f"Current: {loss2}",
                        start_time=SRConfig.start_time,
                    )
                )
                break

    def check_weights_below_threshold(self, logs):
        for layer in self.nnLayers:
            for unit in layer.units:
                for weight in unit.weights:
                    for v, mask in zip(weight.v, weight.mask):
                        if mask == 1 and torch.abs(v) < SRConfig.activeWeightThreshold:
                            logs.append(
                                align_text(
                                    f"!!! weight {weight.name} of unit {unit.name} in layer {layer.id} has a mask of 1 but a value below the threshold: {v}",
                                    start_time=SRConfig.start_time,
                                )
                            )

    def train_nn(
        self,
        learningSteps: int = 0,
        learning_subset_size=1.0,
        learningRate: float = 0.010,
        regularize=True,
        clipWeights: bool = True,
        constrain=True,
        logs=[],
        individualId=-1,
        backpropItersBefore: int = 0,
        lastPruningBackprops: int = 0,
    ):
        """
        :param learningSteps:
        :param learning_subset_size: a proportion of variables to update in each learning iteration
        :param learningRate:
        :param regularize:
        :param clipWeights: True ... weights are clipped to 0, if they have a below-threshold value
        :param constrain:   True ... constraints are considered in the learning process.
        :param logs: a list of strings to which the log messages are appended
        :param individualId:
        :param backpropItersBefore: the owning individual's backprop_iters BEFORE this
            call. backpropItersBefore + learningSteps gives the post-call count used
            to decide periodic pruning.
        :param lastPruningBackprops: the owning individual's backprop_iters at its last
            prune. Pruning fires when (backpropItersBefore + learningSteps) -
            lastPruningBackprops >= SRConfig.pruningStep. The (possibly advanced) value
            is written to self.lastPruningBackprops for the caller to read back.
        :return:
        """

        pt_seed = SRConfig.r.integers(1, 1e6)
        torch.manual_seed(pt_seed)
        eps = 1e-08
        nbOfImprovements = 0

        # --- Expose the pruning watermark on the instance so callers (workers,
        #     main process) can read back the value after train_nn returns -
        #     even via the early return below. Advanced in the pruning block.
        self.lastPruningBackprops = lastPruningBackprops

        # ------------------------------------------------
        # --- Prepare learnable variables
        # ------------------------------------------------
        self.separate_learnable_nonlearnable_params()
        self.get_params_2_learn()
        self.set_forwardpass_masks()

        if not self.params_2_learn:
            # logs.append(
            #     align_text(
            #         "Error in train_nn: no learnable parameters found.",
            #         start_time=SRConfig.start_time,
            #     )
            # )
            return False, 0

        scheduler = None
        optimizer = torch.optim.Adam(self.params_2_learn, lr=learningRate, eps=eps)
        if SRConfig.scheduler["name"]:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, SRConfig.scheduler["name"]
            )
            scheduler = scheduler_class(optimizer, **SRConfig.scheduler["params"])

        # -------------------------------------------------------
        # --- Recalculate valid_loss and rmse_constr
        # -------------------------------------------------------
        self.losses["nbOfActiveNodes"] = self.nbOfActiveNodes
        self.losses["complexity"] = self.complexity
        valid_loss = self.calculateValidationLoss()
        # if self.losses["valid_loss"] > 0 and valid_loss != self.losses["valid_loss"]:
        #     logs.append(
        #         align_text(
        #             f"!!! Recalculated 'valid_loss' for individual {individualId} changed: {self.losses['valid_loss']} --> {valid_loss} \n",
        #             start_time=SRConfig.start_time,
        #         )
        #     )
        self.losses["valid_loss"] = valid_loss
        if constrain:
            c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=0)
            # if self.losses["rmse_constr"] > 0 and c_raw != self.losses["rmse_constr"]:
            #     logs.append(
            #         align_text(
            #             f"!!! Recalculated 'rmse_constr' for individual {individualId} changed: {self.losses['rmse_constr']} --> {c_raw} \n",
            #             start_time=SRConfig.start_time,
            #         )
            #     )
            self.losses["rmse_constr"] = c_raw
        else:
            self.losses["rmse_constr"] = 0

        # ---
        softDominationRMSEValid: list[bsfSubTopology] = []
        softDominationRMSEConstr: list[bsfSubTopology] = []

        bsfNondominatedLosses: dict[str, float] = copy.deepcopy(self.losses)
        bsfNondominatedParams2Learn = copy.deepcopy(self.params_2_learn)
        currSubTopology: bsfSubTopology = bsfSubTopology(
            0, self.losses, self.params_2_learn
        )
        nonDominatedImprovements: list[bsfSubTopology] = [currSubTopology]
        # ---
        self.last_loss_terms = []
        self.last_regularization_terms = []
        # ---
        x_input: torch.tensor
        y_output: torch.tensor
        step = 0

        # --------------------------------------------------------------------------------------
        # --- Choose a set of losses that will be used in the training process
        # --------------------------------------------------------------------------------------
        if learningSteps > SRConfig.nbOfNewbornIterations and learningSteps < 250:
            # --- Use all available loss terms
            if SRConfig.useAllLossTerms:
                # --- Use all available loss terms
                curr_losses = ["train_loss"]
                curr_losses = (
                    curr_losses + ["constr_loss"] if constrain else curr_losses
                )
                curr_losses = curr_losses + ["reg_loss"] if regularize else curr_losses
            else:
                # --- Use a random subset of all available loss terms
                curr_losses = ["train_loss"]
                curr_losses = (
                    curr_losses + ["constr_loss"] if constrain else curr_losses
                )
                k = SRConfig.r.integers(1, len(curr_losses) + 1)
                curr_losses = SRConfig.r.choice(curr_losses, size=k, replace=False)
                curr_losses = curr_losses.tolist()
                curr_losses = curr_losses + ["reg_loss"] if regularize else curr_losses
            # log(
            #     f"Loss terms used: {curr_losses}",
            #     SRConfig.start_time,
            #     level="HIGHLIGHT_MAGENTA",
            # )
        else:
            curr_losses = ["train_loss"]
            curr_losses = curr_losses + ["constr_loss"] if constrain else curr_losses
            curr_losses = curr_losses + ["reg_loss"] if regularize else curr_losses

        # ==============================================================================================================
        # ======= Main loop: learningSteps iterations
        # ==============================================================================================================
        train_nn_StartTime = time.time()
        while step < learningSteps:

            # --------------------------------------------------------------------------------------
            # --- Calculation of loss, regularization, division penalty, and constraints terms
            # --------------------------------------------------------------------------------------
            loss_terms = []
            x_input_data = SRData.x_data
            y_output_data = SRData.y_data

            # --------------------------------------------------------------------------------------
            # --- Calculation of gradients, forward and backward pass
            # --------------------------------------------------------------------------------------
            x_input = torch.tensor(
                x_input_data, requires_grad=True, dtype=torch.float64
            )
            y_output = torch.tensor(y_output_data)
            self.y_hat = self.model_feed_forward(x_input)
            self.updateLossTermMeans: bool
            if (step >= SRConfig.update_step and step % SRConfig.update_step == 0) or (
                step < SRConfig.update_step
            ):
                self.updateLossTermMeans = True
            else:
                self.updateLossTermMeans = False
            # ---------------------------------------------
            # --- Training Loss
            # ---------------------------------------------
            train_mask = torch.ones_like(self.y_hat, dtype=torch.float64)
            self.train_loss = getattr(self, SRConfig.computeTrainValidLoss)(
                self.y_hat * train_mask, y_output, len(SRData.x_data)
            )
            self.last_loss_terms.append(self.train_loss.item())
            if len(self.last_loss_terms) > SRConfig.stack_size:
                self.last_loss_terms = self.last_loss_terms[1:]
            if step % SRConfig.update_step == 0:
                self.last_loss_terms_mean = np.mean(self.last_loss_terms)
            v = round(self.train_loss.item(), SRConfig.precisionDigits)
            if "train_loss" in curr_losses:
                loss_terms.append(self.train_loss)
            # ---------------------------------------------
            # --- Validation Loss
            # ---------------------------------------------
            valid_loss = self.calculateValidationLoss()
            self.losses["valid_loss"] = valid_loss

            # -------------------------------
            # --- Constraint violations
            # -------------------------------
            if constrain:
                c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=step)
                self.losses["rmse_constr"] = c_raw
                if "constr_loss" in curr_losses:
                    loss_terms.extend(self.constraint_terms.values())
                    # for key, term in self.constraint_terms.items():
                    #     log(
                    #         f"constraint_term: {key} / {term.requires_grad}",
                    #         SRConfig.start_time,
                    #         level="DEBUG",
                    #     )

            else:
                self.losses["rmse_constr"] = 0
            # ------------------------------
            # --- Regularization
            # ------------------------------
            if regularize:
                reg_term = self.get_regularization_term(
                    reg_vars=self.params_2_regularize,
                    reg_lambda=1.0,
                    shapes=self.params_2_regularize_shapes,
                )
                self.last_regularization_terms.append(reg_term.item())
                if len(self.last_regularization_terms) > SRConfig.stack_size:
                    self.last_regularization_terms = self.last_regularization_terms[1:]
                if SRConfig.adaptive:
                    b = np.mean(self.last_regularization_terms)
                    if b != 0:
                        # --- Scale so the regularization contribution magnitude is
                        #     approximately reg2rmse * self.train_loss. Use the
                        #     INSTANTANEOUS train_loss (responsive to transients,
                        #     e.g. right after a structural change) divided by the
                        #     smoothed mean raw-reg magnitude 'b' for a stable
                        #     weight. .item() detaches train_loss to a float so the
                        #     weight is a plain constant multiplier and introduces
                        #     no gradient path through train_loss.
                        self.reg_weight = (
                            # --- SRConfig.reg2rmse * self.last_loss_terms_mean / b   
                            SRConfig.reg2rmse * self.train_loss.item() / b   # --- Claude
                        )
                        reg_term *= self.reg_weight
                    else:
                        reg_term *= SRConfig.reg2rmse
                else:
                    reg_term *= SRConfig.reg2rmse
            else:
                reg_term = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            if "reg_loss" in curr_losses:
                loss_terms.append(reg_term)
            # ---
            loss = torch.sum(torch.stack(loss_terms))

            # ---------------------------------------------
            # --- Update best-so-far self.params_2_learn
            # ---------------------------------------------
            firstDominates, secondDominates, firstEqualsSecond = getMutualDominance(
                first=self.losses,
                second=bsfNondominatedLosses,
                criteria=SubTopology.keysPerformance,
            )
            if firstDominates and not firstEqualsSecond:
                softDominationRMSEValid = []
                softDominationRMSEConstr = []
                bsfNondominatedLosses = copy.deepcopy(self.losses)
                bsfNondominatedParams2Learn = copy.deepcopy(self.params_2_learn)
                nonDominatedImprovements.append(
                    bsfSubTopology(step, self.losses, self.params_2_learn)
                )
                nbOfImprovements += 1
            elif self.losses["valid_loss"] < bsfNondominatedLosses["valid_loss"]:
                softDominationRMSEValid.append(
                    bsfSubTopology(
                        step,
                        copy.deepcopy(self.losses),
                        copy.deepcopy(self.params_2_learn),
                    )
                )
            elif self.losses["rmse_constr"] < bsfNondominatedLosses["rmse_constr"]:
                softDominationRMSEConstr.append(
                    bsfSubTopology(
                        step,
                        copy.deepcopy(self.losses),
                        copy.deepcopy(self.params_2_learn),
                    )
                )

            # ---------------------------------------------
            # --- Update weights using the gradients
            # ---------------------------------------------
            try:
                if learning_subset_size == 1.0:
                    # --- update all variables
                    optimizer.zero_grad()

                    if not loss.requires_grad:
                        log(
                            f"loss.requires_grad = {loss.requires_grad}\nloss.grad_fn = {loss.grad_fn}",
                            SRConfig.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                    if loss.grad_fn is None:
                        log(
                            f"loss.grad_fn = {loss.grad_fn}",
                            SRConfig.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                        nbTrue = 0
                        nbFalse = 0
                        for p in self.params_2_learn:
                            if p.requires_grad:
                                nbTrue += 1
                            else:
                                nbFalse += 1
                        log(
                            f"params_2_learn: {len(self.params_2_learn)}, \n\tnbTrue: {nbTrue}, nbFalse: {nbFalse}",
                            SRConfig.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                        log(
                            f"loss.grad_fn is None",
                            SRConfig.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                        raise ValueError("loss.grad_fn is None")

                    loss.backward()
                    optimizer.step()
                    if SRConfig.scheduler["name"]:
                        if isinstance(
                            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

            except Exception as e:
                logging.exception("An error occurred:")
                logs.append(
                    align_text(
                        f"\n\nError in optimizer.apply_gradients(zip(gradients, self.params_2_learn)): {e}\n\n",
                        SRConfig.start_time,
                    )
                )

            # -----------------------------------------------
            # --- Update p.mask based on current p.v
            # -----------------------------------------------
            if clipWeights:
                self.updateLearnableParamsMaskAndValues()

            step = step + 1
        # ==============================================================================================================
        # ======= Main loop: End
        # ==============================================================================================================
        train_nn_EndTime = time.time()
        log(
            f"train_nn() ended, duration: {(train_nn_EndTime - train_nn_StartTime):.4f} seconds",
            SRConfig.start_time,
            file_path=SRConfig.log_file,
            level="HIGHLIGHT_CYAN",
        )

        # ---------------------------------------------------
        # --- Set p.v according to bsfParams2Learn
        # ---------------------------------------------------
        if len(nonDominatedImprovements) > 1:
            # --- HARD non-dominated improvements found
            # logs.append(
            #     align_text(
            #         "Updating subTopology weights based on saved bsfNondominatedParams2Learn.",
            #         start_time=SRConfig.start_time,
            #     )
            # )
            originNonDominatedImprovement = True
            self.nonDominatedImprovement = True
            self.losses["train_loss"] = bsfNondominatedLosses["train_loss"]
            if bsfNondominatedParams2Learn:
                for i, newV in enumerate(bsfNondominatedParams2Learn):
                    self.params_2_learn[i].data.copy_(newV)
                    self.learnableParams[i].set_mask(
                        (newV.abs() >= SRConfig.activeWeightThreshold).float()
                    )
        elif SRConfig.softDomination and (
            len(softDominationRMSEValid) > 0 or len(softDominationRMSEConstr) > 0
        ):
            # --- SOFT improvements found
            originNonDominatedImprovement = False
            self.nonDominatedImprovement = False
            selected_list = None

            logs.append(
                align_text(
                    f"Improvements in softDominationRMSEValid: {len(softDominationRMSEValid)}\n"
                    f"Improvements in softDominationRMSEConstr: {len(softDominationRMSEConstr)}",
                    start_time=SRConfig.start_time,
                )
            )

            if softDominationRMSEValid and softDominationRMSEConstr:
                # --- Both SOFT improvement lists are available
                if "train_loss" in curr_losses and "constr_loss" in curr_losses:
                    selected_list = "valid" if SRConfig.r.random() < 0.5 else "constr"
                else:
                    selected_list = "valid" if "train_loss" in curr_losses else "constr"

            if (selected_list == "valid") or (
                not selected_list
                and softDominationRMSEValid
                and "train_loss" in curr_losses
            ):
                logs.append(
                    align_text(
                        "Updating subTopology weights based on saved softDominationRMSEValid.",
                        start_time=SRConfig.start_time,
                    )
                )
                best_result = min(
                    softDominationRMSEValid, key=lambda x: x.losses["rmse_constr"]
                )
                self.losses["train_loss"] = best_result.losses["train_loss"]
                for i, newV in enumerate(best_result.weights):
                    self.params_2_learn[i].data.copy_(newV)
                    self.learnableParams[i].set_mask(
                        (newV.abs() >= SRConfig.activeWeightThreshold).float()
                    )

            elif (selected_list == "constr") or (
                not selected_list
                and softDominationRMSEConstr
                and "constr_loss" in curr_losses
            ):
                logs.append(
                    align_text(
                        "Updating subTopology weights based on saved softDominationRMSEConstr.",
                        start_time=SRConfig.start_time,
                    )
                )
                best_result = min(
                    softDominationRMSEConstr, key=lambda x: x.losses["valid_loss"]
                )
                self.losses["train_loss"] = best_result.losses["train_loss"]
                for i, newV in enumerate(best_result.weights):
                    self.params_2_learn[i].data.copy_(newV)
                    self.learnableParams[i].set_mask(
                        (newV.abs() >= SRConfig.activeWeightThreshold).float()
                    )

        else:
            originNonDominatedImprovement = False
            # logs.append(
            #     align_text(
            #         "No improvements during train_nn.", start_time=SRConfig.start_time
            #     )
            # )
            self.nonDominatedImprovement = False
            self.losses["train_loss"] = bsfNondominatedLosses["train_loss"]
            if bsfNondominatedParams2Learn:
                for i, newV in enumerate(bsfNondominatedParams2Learn):
                    self.params_2_learn[i].data.copy_(newV)
                    self.learnableParams[i].set_mask(
                        (newV.abs() >= SRConfig.activeWeightThreshold).float()
                    )

        # ---------------------------------------------------
        # --- Update p.mask and p.v based on current p.v
        # ---------------------------------------------------
        if clipWeights:
            self.updateLearnableParamsMaskAndValues()

        somethingWasPruned = False
        # ---------------------------------------------------
        # --- Periodic, backprop-counter-driven pruning.
        # --- Prune when the owning individual has accumulated at least
        # --- SRConfig.pruningStep backprops since its last prune. The watermark
        # --- (self.lastPruningBackprops) is advanced whenever the check fires,
        # --- regardless of whether anything was actually removed, so the cadence
        # --- is one prune attempt per pruningStep backprops.
        # ---------------------------------------------------
        backpropItersAfter = backpropItersBefore + learningSteps
        if (
            SRConfig.pruning
            and (backpropItersAfter - self.lastPruningBackprops) >= SRConfig.pruningStep
        ):
            # --- Prune units that have below-threshold output
            somethingWasPruned = (
                self.pruneUnitsWithBelowThresholdOutput() or somethingWasPruned
            )
            # --- Prune weights that have below-threshold ratio 'abs(w)/max(abs(W))'
            somethingWasPruned = self.pruneWeights_new() or somethingWasPruned
            # --- Advance the watermark to the post-call backprop count
            self.lastPruningBackprops = backpropItersAfter

        # -----------------------------------------------
        # --- Update the number of active nodes
        # -----------------------------------------------
        self.updateActiveNodes()

        # -----------------------------------------------
        # --- RMSE on validation data
        # -----------------------------------------------
        self.losses["valid_loss"] = self.calculateValidationLoss()

        # -------------------------------
        # --- Constraint violations
        # -------------------------------
        c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=step)
        self.losses["rmse_constr"] = c_raw

        return originNonDominatedImprovement, nbOfImprovements

    def updateActiveNodes(self):
        # ----------------------------------------------------------------------------------------------
        # --- Backward pass
        # --- If an active node does not serve as an input to any active node in the subsequent layer then
        # ---   1. the node is inactivated
        # ---   2. all its input weights and masks are set to 0
        # ----------------------------------------------------------------------------------------------
        for l in reversed(range(1, len(self.nnLayers))):  # --- output layer
            currentNodes = torch.zeros(
                (len(self.nnLayers[l - 1].units), 1), dtype=torch.int32
            )  # --- current layer
            for outputNode in self.nnLayers[l].units:  # --- nodes in output layer
                outputNodesInputActivity = torch.tensor(
                    self.activeNodeCoordinates[l - 1], dtype=torch.int32
                )
                outputNodesInputActivity = outputNodesInputActivity.reshape(
                    currentNodes.shape
                )
                for tfvar in outputNode.weights:  # --- for all nodes in output layer
                    if " b" not in tfvar.name:  # --- weights, NOT bias
                        # --- output layer input-nodes activity check
                        newVals = outputNodesInputActivity * tfvar.v
                        tfvar.set_value_mask(newVals)
                        # Accumulate mask values to `currentNodes`
                        currentNodes += tfvar.mask.to(torch.int32)
            # --- current layer activity check
            self.activeNodeCoordinates[l - 1] = (
                torch.logical_and(
                    torch.tensor(self.activeNodeCoordinates[l - 1], dtype=torch.bool),
                    currentNodes.flatten(),
                )
            ).tolist()

            for n, node in enumerate(self.nnLayers[l - 1].units):
                if not self.activeNodeCoordinates[l - 1][n]:
                    for tfvar in node.weights:
                        tfvar.set_allzero_value_mask()
        # ----------------------------------------------------------------------------------------------
        # --- Forward pass
        # --- If an active node has no active input then
        # ---   1. the node is inactivated
        # ---   2. all its input weights and masks are set to 0
        # ----------------------------------------------------------------------------------------------
        inputNodes = torch.ones((self.nnLayers[0].inSize, 1), dtype=torch.int32)
        for l, layer in enumerate(self.nnLayers):
            if l > 0:
                inputNodes = torch.tensor(
                    self.activeNodeCoordinates[l - 1], dtype=torch.int32
                )
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    hasActiveWeights = False
                    for tfvar in node.weights:
                        if " b" not in tfvar.name:  # --- weights, NOT bias
                            if tfvar.hasAnyActiveInput(
                                inputLayerActiveNodes=inputNodes
                            ):  # --- check for active input
                                hasActiveWeights = True
                            else:
                                hasActiveWeights = False
                    if not hasActiveWeights:
                        self.activeNodeCoordinates[l][
                            n
                        ] = False  # --- all weights of this active node are inactive
                        for tfvar in node.weights:
                            tfvar.set_allzero_value_mask()
        # ----------------------------------------------------------------------------------------------
        # --- Calculate the number of active nodes and active weights
        # ----------------------------------------------------------------------------------------------
        self.nbOfActiveNodes = 0
        self.nbOfActiveWeights = 0

        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n] and not ("UnitIdent1" in node.name):
                    self.nbOfActiveNodes += 1
                    for tfvar in node.weights:
                        # if " b" not in tfvar.name:  # --- weights, NOT bias
                        if True:
                            self.nbOfActiveWeights += tfvar.mask.sum().item()

        self.complexity = self.nbOfActiveNodes + self.nbOfActiveWeights / 10000.0

        self.losses["nbOfActiveNodes"] = self.nbOfActiveNodes
        self.losses["complexity"] = self.complexity

    def activateAllInactiveNodes(self):
        # ----------------------------------------------------------------------------------------------
        # --- Forward pass through all hidden layers; output layer node is always active.
        # ---
        # --- For each INACTIVE node (probabilistically gated by SRConfig.perturbation):
        # ---   1. Mark the node as active.
        # ---   2. Randomly initialise ALL of its weights (full random re-init).
        # ---      UnitIdent1 nodes use their structural reset instead of setRandomWeights.
        # ---
        # --- For each already-ACTIVE node (non-UnitIdent1 only):
        # ---   3. For every weight tensor whose original mask (mask_orig) is all-zero
        # ---      (i.e. the weight slot was structurally zero at construction time,
        # ---      e.g. a bias on a unit type that doesn't normally use one, or an
        # ---      entry zeroed by arity constraints), randomise that weight tensor
        # ---      so it gets a chance to become active.
        # ----------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------
        # --- Hidden layers only: activate inactive nodes and fully
        # --- randomise all their weights.
        # ------------------------------------------------------------------
        for l in range(len(self.nnLayers) - 1):
            for n, node in enumerate(self.nnLayers[l].units):
                if not self.activeNodeCoordinates[l][n]:
                    if SRConfig.r.random() >= SRConfig.perturbation:
                        continue
                    self.activeNodeCoordinates[l][n] = True
                    if "UnitIdent1" in node.name:
                        node.reset_weights_and_bias()
                    else:
                        self.nbOfActiveNodes += 1
                        for w in node.weights:
                            if " b" in w.name:  # --- bias weight
                                inputLayerActiveNodes = [True]
                            else:
                                inputLayerActiveNodes = self.nnLayers[l].inSize * [True]
                            w.setRandomWeights(
                                layerId=l,
                                inputLayerActiveNodes=inputLayerActiveNodes,
                                perturbation=False,
                            )

        # ------------------------------------------------------------------
        # --- All layers including output: for every already-active node,
        # --- randomise individual weight positions that are currently zero
        # --- but permitted by mask_orig (i.e. originally-zero weights).
        # --- Skip UnitIdent1 — its zero entries are structural constants.
        # ------------------------------------------------------------------
        for l in range(len(self.nnLayers)):
            for n, node in enumerate(self.nnLayers[l].units):
                if not self.activeNodeCoordinates[l][n]:
                    continue
                if "UnitIdent1" in node.name:
                    continue
                for w in node.weights:
                    w.randomize_originally_zero_weights()

    @staticmethod
    def compareSubtopologies(subtopology1, subtopology2):
        """
        Compares two subtopologies based on their topology. Checks if they have the same nodes active and, if so, if the masks are the same.
        :return: True if both subtopologies have the same topology, False otherwise.
        """
        # check if the number of active nodes is the same
        if subtopology1.nbOfActiveNodes != subtopology2.nbOfActiveNodes:
            return False

        # check if the active nodes are the same
        for layer1, layer2 in zip(
            subtopology1.activeNodeCoordinates, subtopology2.activeNodeCoordinates
        ):
            if layer1 != layer2:
                return False

        # check if the masks are the same
        for layer1, layer2 in zip(subtopology1.nnLayers, subtopology2.nnLayers):
            for unit1, unit2 in zip(layer1.units, layer2.units):
                for weight1, weight2 in zip(unit1.weights, unit2.weights):
                    if not torch.equal(weight1.mask, weight2.mask):
                        return False
                    if not torch.equal(weight1.v, weight2.v):
                        return False

        return True
