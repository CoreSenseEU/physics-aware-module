import warnings

import SRData as SRData

warnings.filterwarnings("ignore")
import torch
import numpy as np
import copy

import SRConfig as SRConfig
from SRUnits import NNUnit, UnitIdent1


class NNLayer:

    def __init__(self, inSize, id):
        self.inSize: int = inSize  # --- nb. of inputs coming from previous layer
        self.id = id
        self.units: list[NNUnit] = []  # --- list of node objects
        self.learnableParams = []

    def add_unit(self, unit):
        learn_params, nolearn_params = unit.define_weight_and_bias_vars(
            layerId=-1, inSize=self.inSize
        )
        nbOfLearnableWeights = 0
        if learn_params is not None:
            self.learnableParams.extend(learn_params)
            for p in learn_params:
                nbOfLearnableWeights += p.mask_orig.shape[0]
        self.units.append(unit)
        return nbOfLearnableWeights

    def clone_layer(self, layerId: int = None, activeNodes: list[list[bool]] = None):
        """
        Clones the layerId-th layer.
        For active nodes:
           - sets the input weights to some values.
           - adds its parameters to newLayer.learnableParams
        For inactive nodes:
           - sets the input weights to zero.
        :param layerId:
        :param activeNodes:
        :return:
        """
        newLayer = NNLayer(self.inSize, self.id)
        # --- Units
        for i, u in enumerate(self.units):
            newUnit = type(u)()

            # --- copy attributes
            newUnit.name = u.name
            newUnit.inId = u.inId
            newUnit.proxyName = u.proxyName
            newUnit.isVariable = u.isVariable
            newUnit.learnable = u.learnable
            newUnit.maxArity = u.maxArity
            newUnit.string_analytic_expression = u.string_analytic_expression
            if hasattr(u, "symbolic_expression") and u.symbolic_expression is not None:
                newUnit.symbolic_expression = copy.deepcopy(u.symbolic_expression)
            newUnit.a = None

            learn_params, nolearn_params = newUnit.define_weight_and_bias_vars(
                layerId=layerId, inSize=newLayer.inSize
            )

            # --- Set input weights and mask of an inactive node to zero
            for w_new, w_old in zip(newUnit.weights, u.weights):
                w_new.set_value_mask(w_old.v)
            if learn_params:
                newLayer.learnableParams.extend(learn_params)
            newLayer.units.append(newUnit)
        # ---
        return newLayer

    @staticmethod
    def create_layer_with_params(
        layerId: int = None,
        layerParams: list = None,
        inputSize: int = None,
        unit_types: dict = None,
        identities: bool = False,
        previousLayer: "NNLayer" = None,
    ):
        """
        Creates a new layer with a given 'layerId'.
        For active nodes:
           - sets the input weights to the given values.
           - adds its parameters to newLayer.learnableParams
        For inactive nodes:
           - sets the input weights to zero.
        :param layerId:
        :param layerParams: a list of tuples in the form (unit.name, unitParams)
        :return: NNLayer
        """
        newLayer = NNLayer(inputSize, layerId)
        # --- Actual units
        for u in unit_types.keys():
            for i in range(unit_types[u]):
                newLayer.add_unit(u())
        # --- Identities
        if identities:
            for i in range(inputSize):
                newUnit = UnitIdent1(i)
                if (layerId == 0) and (i < SRData.SRData.x_data.shape[1]):
                    newUnit.isVariable = True
                # --- current UnitIdent1 points to a previous layer UnitIdent1 that represents a variable
                elif (layerId > 0) and (previousLayer.units[i].isVariable):
                    newUnit.isVariable = True
                newLayer.add_unit(newUnit)

        # -------------------------------------------------------
        # --- Set weights and masks according to 'layerParams'
        # -------------------------------------------------------
        weightsIdx = 0
        for i, u in enumerate(newLayer.units):
            u.a = None
            learn_params, _ = u.define_weight_and_bias_vars(
                layerId=layerId, inSize=newLayer.inSize
            )
            for w_new in u.weights:
                w_new.set_value_mask(layerParams[weightsIdx][1])
                weightsIdx += 1
            if learn_params:
                newLayer.learnableParams.extend(learn_params)
        # ---
        return newLayer

    def randomly_generate_layerNew(
        self, layerId: int = None, layerActiveNodes: list[bool] = None, inputLayer=None
    ):
        """
        Randomly generates the layerId-th layer given the active nodes.
           - Chooses weight configs for the given active nodes.
        :param layerId:
        :param layerActiveNodes: list[bool] ... active nodes of this layer
        :param inputLayer: previous MasterTopology layer of this layer
        :return: newly created newLayer, inputLayerActiveNodes: list[bool]
        """
        # ------------------------------------------------
        # --- Choose weight configs for active nodes
        # ------------------------------------------------
        actualInputLayerActiveNodes = np.zeros((self.inSize, 1), dtype=np.float64)
        inputLayerActiveNodes: list[bool] = self.inSize * [True]
        if sum(inputLayerActiveNodes) == 0:
            inputLayerActiveNodes[SRConfig.r.integers(self.inSize)] = True
        inputLayerActivity = np.array(inputLayerActiveNodes).reshape(
            len(inputLayerActiveNodes), 1
        )
        newLayer = []

        for i, u in enumerate(self.units):
            newUnit = copy.deepcopy(u)
            params = [[] for _ in range(newUnit.nbOfVars)]
            learn_params, _ = newUnit.define_weight_and_bias_vars(
                layerId=layerId, inSize=self.inSize
            )

            if layerActiveNodes[i]:
                weights = []
                for w in newUnit.weights:
                    finalValues = w.clone()

                    if " W" in w.name:
                        if layerId > 0:
                            finalMask = torch.tensor(
                                [
                                    (
                                        int(
                                            SRConfig.r.random()
                                            <= SRConfig.initActiveUnitsProp
                                        )
                                        if el
                                        else 0
                                    )
                                    for el in inputLayerActiveNodes
                                ]
                            ).reshape(inputLayerActivity.shape)

                            for m in range(len(finalMask)):
                                if (
                                    finalMask[m].item() == 1
                                    and inputLayer.units[m].isVariable
                                    and SRConfig.r.random() <= SRConfig.weight_init_one
                                ):
                                    finalValues[m] = (
                                        1.0 if SRConfig.r.random() < 0.5 else -1.0
                                    )
                        else:
                            finalMask = torch.tensor(
                                [
                                    int(SRConfig.r.random() < 0.5) if el else 0
                                    for el in inputLayerActiveNodes
                                ]
                            ).reshape(inputLayerActivity.shape)

                        if torch.sum(finalMask) == 0:
                            nonzero = SRConfig.r.choice(
                                inputLayerActivity.nonzero()[0], 1
                            )[0]
                            finalMask[nonzero] = 1

                        finalMask = w.adjustArity(finalMask)
                        actualInputLayerActiveNodes += finalMask
                        finalValues = finalValues * finalMask

                    weights.append(finalValues.detach().numpy())
            else:
                weights = [np.zeros_like(w.v.detach().numpy()) for w in newUnit.weights]

            newLayer.append((newUnit.name, np.concatenate(weights, axis=0)))

        inputLayerActiveNodes = [el[0] > 0 for el in actualInputLayerActiveNodes]
        return newLayer, inputLayerActiveNodes

    def randomly_generate_layer(
        self, layerId: int = None, layerActiveNodes: list[bool] = None, inputLayer=None
    ):
        """
        Randomly generates the layerId-th layer given the active nodes.
           - Chooses weight configs for the given active nodes.
        :param layerId:
        :param layerActiveNodes: list[bool] ... active nodes of this layer
        :param inputLayer: previous MasterTopology layer of this layer
        :return: newly created newLayer, inputLayerActiveNodes: list[bool]
        """
        # ------------------------------------------------
        # --- Choose weight configs for active nodes
        # ------------------------------------------------
        actualInputLayerActiveNodes: torch.tensor = torch.zeros(
            (self.inSize, 1), dtype=torch.float64
        )
        inputLayerActiveNodes: list[bool] = self.inSize * [True]
        if sum(inputLayerActiveNodes) == 0:
            inputLayerActiveNodes[SRConfig.r.integers(self.inSize)] = True
        inputLayerActivity = np.array(inputLayerActiveNodes).reshape(
            len(inputLayerActiveNodes), 1
        )
        newLayer = NNLayer(self.inSize, self.id)
        for i, u in enumerate(self.units):
            newUnit = copy.deepcopy(u)
            learn_params, _ = newUnit.define_weight_and_bias_vars(
                layerId=layerId, inSize=newLayer.inSize
            )
            if layerActiveNodes[i]:
                # --------------------------------------------------
                # --- Process an active ordinary function newUnit
                # --------------------------------------------------
                if ("UnitIdent1" not in newUnit.name) and (
                    "UnitMultiplySimplified" not in newUnit.name
                ):
                    for w in newUnit.weights:
                        if " W" in w.name:
                            finalValues = w.v.clone()
                            # ----------------------------
                            # --- Choose active weights
                            # ----------------------------
                            nonzero = SRConfig.r.choice(
                                inputLayerActivity.nonzero()[0], 1
                            )[0]
                            if layerId > 0:
                                # --- TODO weights to UnitIdent1 representing variables are set to +/-1 with p==weight_init_one
                                finalMask = torch.tensor(
                                    [
                                        (
                                            int(
                                                SRConfig.r.random()
                                                <= SRConfig.initActiveUnitsProp
                                            )
                                            if el
                                            else 0
                                        )
                                        for el in inputLayerActiveNodes
                                    ]
                                ).reshape(inputLayerActivity.shape)
                                # -----------------------------------------------------------------------
                                # --- Adjust weight leading to UnitIdent1 nodes representing variables
                                # -----------------------------------------------------------------------
                                for m in range(len(finalMask)):
                                    inputUnit: NNUnit = inputLayer.units[m]
                                    if (
                                        (finalMask[m].item() == 1)
                                        and (inputUnit.isVariable)
                                        and (
                                            SRConfig.r.random()
                                            <= SRConfig.weight_init_one
                                        )
                                    ):
                                        if SRConfig.r.random() < 0.5:
                                            newValue = 1.0  # --- TODO   +1
                                        else:
                                            newValue = -1.0  # --- TODO   -1
                                        finalValues[m] = newValue
                            else:
                                # -----------------------------------------------------------------------
                                # --- layerId==0 function nodes choose their input variables with p==0.5
                                # -----------------------------------------------------------------------
                                finalMask = torch.tensor(
                                    [
                                        int(SRConfig.r.random() < 0.5) if el else 0
                                        for el in inputLayerActiveNodes
                                    ]
                                ).reshape(inputLayerActivity.shape)
                            if torch.sum(finalMask) == 0:
                                finalMask[nonzero] = 1
                            finalMask = w.adjustArity(finalMask)
                            actualInputLayerActiveNodes = (
                                actualInputLayerActiveNodes + finalMask
                            )
                            # -----------------------------------
                            # --- Set inactive weights to zero
                            # -----------------------------------
                            finalValues = finalValues * finalMask
                        else:
                            finalValues = w.v.clone()
                        # ---------------------------
                        # --- Set finalValues to w
                        # ---------------------------
                        w.set_value_mask(finalValues)
                # -------------------------------------------------
                # --- Process an active UnitIdent1 in layerId==0
                # -------------------------------------------------
                elif ("UnitIdent1" in newUnit.name) and (layerId == 0):
                    newUnit: UnitIdent1
                    if newUnit.inId < SRData.SRData.x_data.shape[1]:
                        newUnit.isVariable = True
                # ---
                if learn_params:
                    newLayer.learnableParams.extend(learn_params)
            else:
                # -------------------------------------------------------------
                # --- Inactive node
                # --- Set input weights and mask of an inactive node to zero.
                # -------------------------------------------------------------
                for w in newUnit.weights:
                    w.set_allzero_value_mask()
            newLayer.units.append(newUnit)
        # ---
        inputLayerActiveNodes = [el[0] > 0 for el in actualInputLayerActiveNodes]
        return newLayer, inputLayerActiveNodes

    def forward_pass(self, x_data):
        a = []
        z = []
        for u in self.units:
            a_i, z_i = u.forward_pass(x_data)
            a.append(a_i)
            if z_i is not None:
                z.append(z_i)
        a = torch.cat(a, 1)
        return a, z
