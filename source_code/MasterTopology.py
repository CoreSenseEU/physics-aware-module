import heapq
import itertools
import sys
import numpy as np
import torch

from dataclasses import dataclass
from collections import namedtuple

import SRConfig as SRConfig
import SRData as SRData
from NNLayer import NNLayer
from SubTopology import SubTopology
from InputConfiguration import InputConfiguration
from SRUnits import UnitIdent1, MyTorchVariable
from prefix import prefix_to_tree, Node
from logger import log
from utils import NNTopology


@dataclass
class SubTree:
    tree_node: Node  # root of the subtree
    layer: int  # layer id of the current operator
    unit: int  # unit id of the current operator
    tree_branch: str  # L if left subtree, R if right subtree


NodePosition = namedtuple("NodePosition", ["layer", "unit"])


def extractConstantFromSubtree(left, right):
    """
    Extracts a constant value and updates the tree node based on the input nodes.

    :param left: The left child node.
    :param right: The right child node.
    :return: A tuple containing the constant value and the updated tree node.
    """
    try:
        return float(left.value), right
    except (ValueError, TypeError):
        pass

    try:
        return float(right.value), left
    except (ValueError, TypeError):
        pass

    if left.value == "U":
        try:
            return -float(left.left.value), right
        except (ValueError, TypeError, AttributeError):
            pass

    if right.value == "U":
        try:
            return -float(right.left.value), left
        except (ValueError, TypeError, AttributeError):
            pass

    return 0.0, None


def checkConfigShape(
    nodeWeights: list[MyTorchVariable], configWeights: list[np.ndarray]
):
    for w, c in zip(nodeWeights, configWeights):
        if w.v.shape != c.shape:
            return False
    return True


class ExprNode:
    def __init__(self, op_or_val, children=None):
        self.op_or_val = op_or_val
        self.children = children or []


def parse_prefix(tokens: list[str]) -> ExprNode:
    token = tokens.pop(0)
    if token in {"+", "-", "*", "/"}:
        left = parse_prefix(tokens)
        right = parse_prefix(tokens)
        return ExprNode(token, [left, right])
    else:
        return ExprNode(token)


class NodeUtilizations:
    def __init__(self, nnLayers: list[NNLayer]):
        self.k = (
            0.0  # --- the number of subTopologies collected in the nodeUtilizations
        )
        self.nodeUtilizationsAbs: list[list[int]] = (
            []
        )  # --- absolute utilization values for each node
        self.nodeUtilizationsRel: list[list[float]] = (
            []
        )  # --- utilization value from interval [0, 1], i.e., a proportion
        # --- of active node occurences in the population
        for layer in nnLayers:
            self.nodeUtilizationsAbs.append(len(layer.units) * [0])
            self.nodeUtilizationsRel.append(len(layer.units) * [0.0])

    def reset(self):
        self.k = 0.0
        newUtilsAbs = []
        newUtilsRel = []
        for layer in self.nodeUtilizationsAbs:
            newUtilsAbs.append(len(layer) * [0])
            newUtilsRel.append(len(layer) * [0.0])
        self.nodeUtilizationsAbs = newUtilsAbs
        self.nodeUtilizationsRel = newUtilsRel

    def updateNodeUtilizations(self, activeNodes: list[list[bool]]):
        self.k += 1.0
        for l, layer in enumerate(activeNodes):
            for n, node in enumerate(layer):
                self.nodeUtilizationsAbs[l][n] += activeNodes[l][n]
                self.nodeUtilizationsRel[l][n] = self.nodeUtilizationsAbs[l][n] / self.k


class MasterTopology:
    translations = {
        "m": ["UnitMultiply", "UnitMultiply_b"],
        "/": ["UnitDivide", "UnitDivide_b"],
        "+": ["UnitIdent"],
        "sin": ["UnitSin", "UnitSin_b"],
        "cos": ["UnitCos", "UnitCos_b"],
        "tanh": ["UnitTanh", "UnitTanh_b"],
        "arctan": ["UnitArcTan", "UnitArcTan_b"],
        "**2": [
            "UnitSquare",
            "UnitSquare_b",
        ],
        "sqrt": ["UnitSqrt", "UnitSqrt_b"],
        "**3": ["UnitCube", "UnitCube_b"],
        "*": ["UnitIdent"],  # just a weight
        "B": [
            "UnitIdent"
        ],  # just a bias (if it was bias of operator it would be after the operator itself)
    }

    def __init__(self, inputSize=0, layer_defs=None, identities=None):
        self.nnLayers: list[NNLayer] = (
            []
        )  # --- outer list of layers, inner list of layer'nodes
        self.availableOperators = self.getAvailableOperators()
        # MasterTopology.networkHistory = []
        # --- Forward pass
        for l in range(len(layer_defs)):
            layer_i = self.create_layer(
                l, in_size=inputSize, unit_types=layer_defs[l], identities=identities[l]
            )
            inputSize = len(layer_i.units)
        self.nodeUtilizations = NodeUtilizations(self.nnLayers)

    def getAvailableOperators(self):
        reverse_map = {
            val: key
            for key, values in MasterTopology.translations.items()
            for val in values
        }
        used_operators = set()
        used_operators.update({"+", "B", "U", "*"})  # always available

        for layer in self.nnLayers:
            for unit in layer.units:
                if unit.name in reverse_map:
                    used_operators.add(reverse_map[unit.name])

        return used_operators

    def updateNodeUtilizations(self, activeNodes: list[list[bool]]):
        self.nodeUtilizations.updateNodeUtilizations(activeNodes)

    def resetNodeUtilizations(self):
        self.nodeUtilizations.reset()

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
                if (layerId == 0) and (i < SRData.SRData.x_data.shape[1]):
                    newUnit.isVariable = True
                elif (layerId > 0) and (
                    self.nnLayers[layerId - 1].units[i].isVariable
                ):  # --- current UnitIdent1 points to a previous layer UnitIdent1 that represents a variable
                    newUnit.isVariable = True
                nnLayer.add_unit(newUnit)
        # ---
        self.nnLayers.append(nnLayer)
        return nnLayer

    def generateRandomSubTopology(self):
        """
        Generates randomly a subTopology starting from the output layer and proceeding backwards.
        Backward pass, from the output layer to the first one.
        :return: A clone of the MasterTopology with its weights set.
        """
        subTopology = SubTopology()
        subTopology.activeNodeCoordinates = []
        # ---
        activeNodes: list[bool] = [True]
        # -------------------------------------------------
        # --- Backward pass, start with the output layer
        # -------------------------------------------------
        for l in reversed(range(len(self.nnLayers))):
            subTopology.activeNodeCoordinates.insert(0, activeNodes)
            if l > 0:
                prevLayer = self.nnLayers[l - 1]
            else:
                prevLayer = None
            newLayer, activeNodes = self.nnLayers[l].randomly_generate_layer(
                layerId=l, layerActiveNodes=activeNodes, inputLayer=prevLayer
            )
            subTopology.learnableParams.extend(newLayer.learnableParams)
            subTopology.nnLayers.insert(0, newLayer)
        # ---------------------------------------------------
        # --- Rectify connectivity and update active nodes
        # ---------------------------------------------------
        subTopology.setProxyNames()
        # self.rectifyActiveNodesConnectivity(subTopology)

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Redirect input links to inactive nodes to active nodes
        # --------------------------------------------------------------------
        subTopology = self.correction_redirecting_links_to_inactive_nodes(subTopology)

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Inactive active nodes with no output link
        # --------------------------------------------------------------------
        subTopology, _ = self.correction_resolve_active_nodes_without_output_link(
            subTopology
        )

        subTopology.updateActiveNodes()
        # ---
        return subTopology

    def generate_random_subtopology_limited_arity(self):
        """
        Generates randomly a subTopology in a forward manner.
        :return: A new subTopology with randomly initialized weights and respecting the max arity constraints.
        """
        # log(f"generate_random_subtopology_limited_arity starts ...", level="HIGHLIGHT_MAGENTA")
        subTopology = SubTopology()
        subTopology.activeNodeCoordinates = []
        # -------------------------------------------------
        # --- Forward pass, start with the input layer
        # -------------------------------------------------
        offspringActiveNodes: list[list[bool]] = []
        inputSize = SRData.SRData.x_data.shape[1]
        for l, layer in enumerate(self.nnLayers):
            layer_l = subTopology.create_layer(
                l,
                in_size=inputSize,
                unit_types=NNTopology.all_layer_defs[0][l],
                identities=NNTopology.all_identities[0][l],
            )

            activeNodesLayer: list[bool] = []
            # ---
            for n, node in enumerate(layer_l.units):
                currValue: bool = True if SRConfig.r.random() < 1.0 else False
                activeNodesLayer.append(currValue)
                if currValue:
                    if "UnitIdent1" in node.name:
                        # --- UnitIdent1 node: set its default
                        node.reset_weights_and_bias()
                    else:
                        # --- Ordinary unit: generate random values
                        for w in node.weights:
                            w, inputLayerActiveNodes = (
                                self.generate_random_weights_with_max_arity(
                                    w,
                                    max_arity=SRConfig.max_arities.get(
                                        node.name, sys.maxsize
                                    ),
                                    layer_id=l,
                                    inputSize=layer_l.inSize,
                                    completeInputLayerActiveNodes=(
                                        offspringActiveNodes[l - 1] if l > 0 else None
                                    ),
                                )
                            )
                # ----------------------------------------------------
                # --- It's an inactive node, set all values to zero
                # ----------------------------------------------------
                else:
                    for w in node.weights:
                        w.set_allzero_value_mask()

            inputSize = len(layer_l.units)

            # log(f"active nodes in layer {l}: {activeNodesLayer}", level="HIGHLIGHT_GREEN")
            offspringActiveNodes.append(activeNodesLayer)
            # log(f"active nodes in layer {l}: {offspringActiveNodes[l]}", level="HIGHLIGHT_CYAN")
        subTopology.activeNodeCoordinates = offspringActiveNodes
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[0]}", level="HIGHLIGHT_GREEN")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[1]}", level="HIGHLIGHT_GREEN")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[2]}", level="HIGHLIGHT_GREEN")

        # ---------------------------------------------------
        # --- Rectify connectivity and update active nodes
        # ---------------------------------------------------
        subTopology.setProxyNames()

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Redirect input links to inactive nodes to active nodes
        # --------------------------------------------------------------------
        subTopology = self.correction_redirecting_links_to_inactive_nodes(subTopology)

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Inactive active nodes with no output link
        # --------------------------------------------------------------------
        subTopology, _ = self.correction_resolve_active_nodes_without_output_link(
            subTopology
        )

        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[0]}", level="HIGHLIGHT_GREEN")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[1]}", level="HIGHLIGHT_GREEN")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[2]}", level="HIGHLIGHT_GREEN")

        subTopology.updateActiveNodes()
        # ---
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[0]}", level="HIGHLIGHT_RED")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[1]}", level="HIGHLIGHT_RED")
        # log(f"generate_random_subtopology_limited_arity, active nodes: {subTopology.activeNodeCoordinates[2]}", level="HIGHLIGHT_RED")
        # log(f"generate_random_subtopology_limited_arity ended with complexity: {subTopology.complexity}", level="HIGHLIGHT_RED")
        return subTopology

    def findPosOfUnit(
        self,
        node: Node,
        subTopology: SubTopology,
        occupied: list[list[str]],
        layer: int,
    ):
        """
        Finds the position of the unit in the given layer.

        :returns: SubTree to put in the queue for further processing and the position of the unit in the layer to assign weight to
        """

        def propagate_to_higher_layers(
            start_layer: int, end_layer: int, nodeToConnect: int, prefix: str
        ):
            while start_layer <= end_layer:
                foundConnectingUnit = False
                for i, unit in enumerate(subTopology.nnLayers[start_layer].units):
                    if unit.name == "UnitIdent1" and unit.inId == nodeToConnect:
                        foundConnectingUnit = True
                        unit.reset_weights_and_bias()
                        subTopology.activeNodeCoordinates[start_layer][i] = True
                        occupied[start_layer][i] = prefix
                        nodeToConnect = i
                        break
                if (
                    not foundConnectingUnit
                ):  # unable to propagate the unit to higher layers
                    return -1
                start_layer += 1
            return nodeToConnect

        if node.value.startswith("x") and node.value[1:].isdigit():  # ident
            toConnect = propagate_to_higher_layers(
                0, layer, int(node.value[1:]), node.prefix
            )
            return None, toConnect

        else:
            # check if wanted unit is already created
            searchFor = node.value
            if node.value == "**":
                log(
                    f'findPosOfUnit(): node.value == "**" should not appear here any more.',
                    start_time=SRConfig.start_time,
                    file_path=SRConfig.log_file,
                    level="ERROR",
                )
                if node.right.value == 0.5:
                    searchFor = "sqrt"
                elif node.right.value == 3.0:
                    searchFor = "**3"
                elif node.right.value == 2.0:
                    searchFor = "**2"
            depth = node.depth
            while depth <= layer:
                for i, unit in enumerate(subTopology.nnLayers[depth].units):
                    if unit.name in MasterTopology.translations[searchFor]:
                        if occupied[depth][i] == node.prefix:
                            if (
                                depth == layer
                            ):  # no subtree to process, no connecting unit
                                return None, i
                            else:  # no subtree to process, assign weights to the last connecting unit
                                toConnect = propagate_to_higher_layers(
                                    depth + 1, layer, i, node.prefix
                                )
                                return None, toConnect
                depth += 1

            # if not found, occupy unit starting from the most shallow layer
            depth = node.depth
            while depth <= layer:
                for i, unit in enumerate(subTopology.nnLayers[depth].units):
                    if unit.name in MasterTopology.translations[searchFor]:
                        if occupied[depth][i] == "":
                            occupied[depth][i] = node.prefix
                            if depth == layer:
                                return NodePosition(layer, i), i
                            else:
                                toConnect = propagate_to_higher_layers(
                                    depth + 1, layer, i, node.prefix
                                )
                                return NodePosition(depth, i), toConnect
                depth += 1

        return None, -1

    def ExprToSubTopology(self, expr: str):
        subTopology = self.generateRandomSubTopology()
        subTopology.resetSubTopology()
        occupied = [
            ["" for _ in range(len(self.nnLayers[l].units))]
            for l in range(len(self.nnLayers))
        ]

        tree_node = prefix_to_tree(expr)
        l = len(self.nnLayers) - 1

        curr_state = SubTree(tree_node, l, 0, "L")
        occupied[l][0] = tree_node.prefix
        # priority: (-depth, layer, counter)
        counter = itertools.count()
        to_process = [((-tree_node.depth, l, next(counter)), curr_state)]  # min-heap
        heapq.heapify(to_process)

        while to_process:
            _, curr_state = heapq.heappop(to_process)
            tree_node = curr_state.tree_node
            left = tree_node.left
            right = tree_node.right

            if tree_node.value == "*":
                constant, tree_node = extractConstantFromSubtree(left, right)

                next_layer = curr_state.layer - 1
                if next_layer < 0:
                    nodeToAssignWeight = int(
                        tree_node.value[1:]
                    )  # extract variable position
                else:
                    next_pos, nodeToAssignWeight = self.findPosOfUnit(
                        tree_node, subTopology, occupied, next_layer
                    )
                    if nodeToAssignWeight == -1:  # error
                        log(
                            f"ExprToSubTopology(): {expr}\n\tself.findPosOfUnit(tree_node = '{tree_node.value}') failed.",
                            start_time=SRConfig.start_time,
                            file_path=SRConfig.log_file,
                            level="ERROR",
                        )
                        return None
                    if not tree_node.value.startswith("x") and next_pos:
                        heapq.heappush(
                            to_process,
                            (
                                (-tree_node.depth, next_pos.layer, next(counter)),
                                SubTree(tree_node, next_pos.layer, next_pos.unit, "L"),
                            ),
                        )

                if curr_state.tree_branch == "L":
                    curr_weight = 0
                elif (
                    "b1"
                    in subTopology.nnLayers[curr_state.layer]
                    .units[curr_state.unit]
                    .weights[1]
                    .name
                ):
                    curr_weight = 2
                else:
                    curr_weight = 1

                to_assign = (
                    subTopology.nnLayers[curr_state.layer]
                    .units[curr_state.unit]
                    .weights[curr_weight]
                    .v.clone()
                )
                # log(
                #     f"len(to_assign): {len(to_assign)}",
                #     start_time=SRConfig.start_time,
                #     file_path=SRConfig.log_file,
                #     level="HIGHLIGHT_MAGENTA",
                # )

                to_assign[nodeToAssignWeight] = constant
                subTopology.nnLayers[curr_state.layer].units[curr_state.unit].weights[
                    curr_weight
                ].set_value_mask(torch.tensor(to_assign, dtype=torch.float64))
                subTopology.activeNodeCoordinates[curr_state.layer][
                    curr_state.unit
                ] = True

            elif tree_node.value == "B":
                constant, tree_node = extractConstantFromSubtree(left, right)
                if curr_state.tree_branch == "L":
                    curr_weight = 1
                else:
                    curr_weight = 3

                subTopology.nnLayers[curr_state.layer].units[curr_state.unit].weights[
                    curr_weight
                ].set_value_mask(torch.tensor([[constant]], dtype=torch.float64))

                heapq.heappush(
                    to_process,
                    (
                        (-left.depth, curr_state.layer, next(counter)),
                        SubTree(
                            left,
                            curr_state.layer,
                            curr_state.unit,
                            curr_state.tree_branch,
                        ),
                    ),
                )

            else:
                if left:
                    heapq.heappush(
                        to_process,
                        (
                            (-left.depth, curr_state.layer, next(counter)),
                            SubTree(left, curr_state.layer, curr_state.unit, "L"),
                        ),
                    )

                if right:
                    branch = (
                        "L" if tree_node.value == "+" else "R"
                    )  # ident node is special case
                    heapq.heappush(
                        to_process,
                        (
                            (-right.depth, curr_state.layer, next(counter)),
                            SubTree(right, curr_state.layer, curr_state.unit, branch),
                        ),
                    )

        subTopology.updateActiveNodes()

        return subTopology

    def subTopologyCrossoverB(
        self, subTopologyA: SubTopology = None, subTopologyB: SubTopology = None
    ):
        """
        Forward pass!
        Inherits activeNodeCoordinates[l][n] either from subTopologyA or from subTopologyB.
        Weights are inherited with a probability SRConfig.inheritWeights.
        Otherwise they are taken from an existing config, if applicable, or randomly regenerated anew.
        :param subTopologyA:
        :param subTopologyB:
        :return: offspring ... a new subTopology as a combinantion of the two parents
                 changed ... True, if the offspring differs from both parents. False, otherwise.
        """
        res: SubTopology = subTopologyA.cloneSubTopology()
        # ---
        pExchange = 0.75
        activeNodes = []
        inheritedFrom = [[-1 for _ in layer.units] for layer in res.nnLayers]

        for l, layer in enumerate(res.nnLayers):
            activeNodesLayer = []
            # ---
            for n, node in enumerate(layer.units):
                if SRConfig.r.random() < pExchange:
                    # -------------------------
                    # --- Use subTopologyA
                    # -------------------------
                    currValue = subTopologyA.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    # ---
                    if currValue:
                        inheritedFrom[l][n] = 2
                        if (
                            l < len(res.nnLayers) - 1
                        ) and SRConfig.r.random() > SRConfig.inheritWeights:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            # --- UnitIdent1 node: set its default
                            if "UnitIdent1" in node.name:
                                node.reset_weights_and_bias()
                            else:
                                # --------------------------------------------
                                # --- Ordinary unit: generate random values
                                # --------------------------------------------
                                for w in node.weights:
                                    inputLayerActiveNodes: list[bool]
                                    if " b" in w.name:  # --- bias
                                        inputLayerActiveNodes = [True]
                                    elif l == 0:
                                        inputLayerActiveNodes = layer.inSize * [True]
                                    else:
                                        # inputLayerActiveNodes = subTopologyA.activeNodeCoordinates[l-1]
                                        inputLayerActiveNodes = activeNodes[l - 1]
                                    w.setRandomWeights(
                                        layerId=l,
                                        inputLayerActiveNodes=inputLayerActiveNodes,
                                    )
                        else:
                            inheritedFrom[l][n] = 1
                    else:
                        # ----------------------------------------------------
                        # --- It's an inactive node, set all values to zero
                        # ----------------------------------------------------
                        for w in node.weights:
                            w.set_allzero_value_mask()
                else:
                    # -----------------------
                    # --- Inherit from B
                    # -----------------------
                    currValue = subTopologyB.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    # --- Change node weights to the weights of the second parent, subTopologyB
                    if not currValue:
                        for w in node.weights:
                            w.set_allzero_value_mask()
                    else:
                        inheritedFrom[l][n] = 2
                        if (
                            l == len(res.nnLayers) - 1
                        ) or SRConfig.r.random() < SRConfig.inheritWeights:
                            # --- INHERIT; use the old subTopologyB weights, if applicable
                            # inheritedFrom[l][n] = 2
                            for wA, wB in zip(
                                node.weights, subTopologyB.nnLayers[l].units[n].weights
                            ):
                                if wA.mask.shape == wB.mask.shape:
                                    wA.set_value_mask(wB.v)
                        else:
                            # --------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # --------------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    inputLayerActiveNodes: list[bool]
                                    if " b" in w.name:  # --- bias
                                        inputLayerActiveNodes = [True]
                                    elif l == 0:
                                        inputLayerActiveNodes = layer.inSize * [True]
                                    else:
                                        # inputLayerActiveNodes = subTopologyA.activeNodeCoordinates[l-1]
                                        inputLayerActiveNodes = activeNodes[l - 1]
                                    w.setRandomWeights(
                                        layerId=l,
                                        inputLayerActiveNodes=inputLayerActiveNodes,
                                    )
            activeNodes.append(activeNodesLayer)
        res.activeNodeCoordinates = activeNodes
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(res.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(res, l, n)
        # ---
        return res, inheritedFrom

    def subTopology_crossover_limited_arity(
        self, subTopologyA: SubTopology = None, subTopologyB: SubTopology = None
    ):
        """
        Forward pass!
        Takes into account maximum arity of units!
        Inherits activeNodeCoordinates[l][n] either from subTopologyA or from subTopologyB.
        Weights are inherited with a probability SRConfig.inheritWeights.
        Otherwise they are taken from an existing config, if applicable, or randomly regenerated anew.
        :param subTopologyA:
        :param subTopologyB:
        :return: res ... a new subTopology as a combination of the two parents
                 inheritedFrom ... layer-wise information about the origin of each node (1 for subTopologyA, 2 for subTopologyB, -1 if not inherited).
        """

        # -----------------------------------------------------------
        # --- Default: Use subTopologyA
        # -----------------------------------------------------------
        res: SubTopology = subTopologyA.cloneSubTopology()

        offspringActiveNodes: list[list[bool]] = []
        # ---
        pExchange = 0.75  # --- probability of inheriting active node from subTopologyA (vs subTopologyB)
        # --- inheritedFrom: -1 if not inherited, 1 if the node is inherited from A, 2 if the node is inherited from B
        inheritedFrom: list[list[int]] = [
            [-1 for _ in layer.units] for layer in res.nnLayers
        ]

        for l, layer in enumerate(res.nnLayers):
            activeNodesLayer: list[bool] = []
            # ---
            for n, node in enumerate(layer.units):
                if SRConfig.r.random() < pExchange:
                    # -----------------------------------------------------------
                    # --- Use subTopologyA
                    # -----------------------------------------------------------
                    currValue = subTopologyA.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    if currValue:
                        inheritedFrom[l][n] = 1
                        if (
                            l < len(res.nnLayers) - 1
                        ) and SRConfig.r.random() > SRConfig.inheritWeights:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                # log(f"A1) UnitIdent1 node ({node.name} / {node.proxyName} / ({l},{n})): set its default 'node.inId={node.inId}'")
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    w, inputLayerActiveNodes = (
                                        self.generate_random_weights_with_max_arity(
                                            w,
                                            max_arity=SRConfig.max_arities.get(
                                                node.name, sys.maxsize
                                            ),
                                            layer_id=l,
                                            inputSize=layer.inSize,
                                            completeInputLayerActiveNodes=(
                                                offspringActiveNodes[l - 1]
                                                if l > 0
                                                else None
                                            ),
                                        )
                                    )
                        else:
                            # ----------------------------------------
                            # --- Inherit weights from subTopologyA
                            # ----------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                # log(f"A2) UnitIdent1 node ({node.name} / {node.proxyName} / ({l},{n})): set its default 'node.inId={node.inId}'")
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    if " b" not in w.name:
                                        w.adjust_value_mask_for_max_arity()
                    else:
                        # ----------------------------------------------------
                        # --- It's an inactive node, set all values to zero
                        # ----------------------------------------------------
                        for w in node.weights:
                            w.set_allzero_value_mask()
                else:
                    # ----------------------------------------------------------
                    # --- Use subTopologyB
                    # ----------------------------------------------------------
                    currValue = subTopologyB.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    if not currValue:
                        # --- It's an inactive node, set all values to zero
                        for w in node.weights:
                            w.set_allzero_value_mask()
                    else:
                        inheritedFrom[l][n] = 2
                        if (
                            l == len(res.nnLayers) - 1
                        ) or SRConfig.r.random() < SRConfig.inheritWeights:
                            # ------------------------------------------------------------------
                            # --- INHERIT weights from subTopologyB,
                            # --- if subTopologyA and subTopologyB have the same active nodes
                            # --- in the previous layer
                            # ------------------------------------------------------------------
                            # if "UnitIdent1" in node.name:
                            # log(f"B1) UnitIdent1 node ({node.name} / {node.proxyName} / ({l},{n})): set its default 'node.inId={node.inId}'")
                            for wA, wB in zip(
                                node.weights, subTopologyB.nnLayers[l].units[n].weights
                            ):
                                if wA.mask.shape == wB.mask.shape:
                                    wA.set_value_mask(wB.v)
                                    wA.adjust_value_mask_for_max_arity()
                        else:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                # log(f"B2) UnitIdent1 node ({node.name} / {node.proxyName} / ({l},{n})): set its default 'node.inId={node.inId}'")
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    # if l == 0:
                                    # log(f"[{l}, {n}], {w.name}.mask before: {w.mask.numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
                                    # log(f"[{l}, {n}], {w.name}.v before: {w.v.detach().cpu().numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
                                    w, inputLayerActiveNodes = (
                                        self.generate_random_weights_with_max_arity(
                                            w,
                                            max_arity=SRConfig.max_arities.get(
                                                node.name, sys.maxsize
                                            ),
                                            layer_id=l,
                                            inputSize=layer.inSize,
                                            completeInputLayerActiveNodes=(
                                                offspringActiveNodes[l - 1]
                                                if l > 0
                                                else None
                                            ),
                                        )
                                    )
                                    # if l == 0:
                                    #     log(f"[{l}, {n}], {w.name}.mask after: {w.mask.numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
                                    #     log(f"[{l}, {n}], {w.name}.v after: {w.v.detach().cpu().numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
            # log(f"active nodes in layer {l}: {activeNodesLayer}", level="HIGHLIGHT_GREEN")
            offspringActiveNodes.append(activeNodesLayer)
            # log(f"active nodes in layer {l}: {offspringActiveNodes[l]}", level="HIGHLIGHT_CYAN")
        res.activeNodeCoordinates = offspringActiveNodes
        # log(f"active nodes in layer {0}: {res.activeNodeCoordinates[0]}", level="HIGHLIGHT_MAGENTA")

        # --------------------------------------------------------------------
        # --- Network review
        # --------------------------------------------------------------------
        # for l, active_nodes_in_layer in enumerate(res.activeNodeCoordinates):
        #     log(f"active nodes in layer {l}: {np.where(active_nodes_in_layer)[0]}")
        #     log(f"active nodes in layer {l}: {active_nodes_in_layer}", level="HIGHLIGHT_GREEN")
        # for w in res.nnLayers[-1].units[0].weights:
        #     wlist = [f"({el1}, {el2})" for el1, el2 in zip(range(len(w.mask.numpy().copy().tolist())), w.mask.numpy().copy().flatten().tolist())]
        #     log(f"\toutput node mask: {wlist}")

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Redirect input links to inactive nodes to active nodes
        # --------------------------------------------------------------------
        res = self.correction_redirecting_links_to_inactive_nodes(res)

        # --------------------------------------------------------------------
        # --- Backward–pass correction of the neural network topology
        # --- Inactive active nodes with no output link
        # --------------------------------------------------------------------
        res, inheritedFrom = self.correction_resolve_active_nodes_without_output_link(
            res, inheritedFrom
        )

        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(res.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(res, l, n)
        # ---
        return res, inheritedFrom

    def correction_redirecting_links_to_inactive_nodes(self, st: SubTopology):
        """
        Backward pass, from the output layer to the first one.
        For each layer, find all inactive nodes that are used in some node of the next layer and replace all output links leading from these inactive nodes with output links leading from active nodes of the same layer.
        """
        for l, layer in reversed(list(enumerate(st.nnLayers))):
            if l == 0:
                continue  # --- do not check the first hidden layer
            prevLayerActiveNodes = st.activeNodeCoordinates[l - 1]
            # -------------------------------------------------------------------------------------
            # --- Find all inactive nodes of layer 'l-1' that are used in some node of layer 'l'
            # -------------------------------------------------------------------------------------
            inactiveButUsedNodes = np.zeros(
                len(prevLayerActiveNodes), dtype=bool
            ).flatten()
            # log(f"inactiveButUsedNodes: {inactiveButUsedNodes}")
            for n, node in enumerate(layer.units):
                for w in node.weights:
                    if " b" not in w.name:
                        # --- Check, if all node's input links lead from active nodes
                        curr_mask = w.mask.numpy().copy().flatten()
                        inactiveUsedPerNode = (curr_mask == 1) & (
                            ~np.array(prevLayerActiveNodes)
                        )
                        # log(f"curr_mask: {curr_mask}")
                        # log(f"prevLayerActiveNodes: {prevLayerActiveNodes}")
                        # log(f"inactiveUsedPerNode: {inactiveUsedPerNode}")
                        inactiveUsedPerNode = inactiveUsedPerNode.astype(
                            bool, copy=False
                        )
                        inactiveButUsedNodes |= inactiveUsedPerNode
            # -------------------------------------------------------------------------------------
            # --- Replace all output links leading from the inactiveButUsedNodes of layer 'l-1'
            # --- with output links leading from active nodes of layer 'l-1'.
            # -------------------------------------------------------------------------------------
            # log(f"inactiveButUsedNodes ({type(inactiveButUsedNodes)}, {np.sum(inactiveButUsedNodes)}), l={l}: {inactiveButUsedNodes}")
            for n, node in enumerate(layer.units):
                for w in node.weights:
                    if " b" not in w.name:
                        # --- Check, if the node uses some of the inactiveButUsedNodes
                        curr_mask = w.mask.numpy().copy().flatten()
                        inputToRedirect = (curr_mask == 1) & (
                            np.array(inactiveButUsedNodes)
                        )
                        # --- Redirect node inputs from inactiveButUsedNodes to active nodes
                        if np.sum(inputToRedirect) > 0:
                            # --- Redirect to a random active node
                            # log(f"prevLayerActiveNodes: {prevLayerActiveNodes}", level="HIGHLIGHT_MAGENTA")
                            # log(f"w.mask before: {w.mask.numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
                            w.adjust_weights_to_prev_layer_active_nodes(
                                prevLayerActiveNodes
                            )
                            # log(f"w.mask after: {w.mask.numpy().flatten()}", level="HIGHLIGHT_MAGENTA")
        return st

    def correction_resolve_active_nodes_without_output_link(
        self, st: SubTopology, inheritedFrom: list[list[int]] = None
    ):
        """
        Backward pass, from the output layer to the first one.
        """
        updated_active_nodes: list[list[bool]] = [
            layer.copy() for layer in st.activeNodeCoordinates
        ]

        for l, layer in reversed(list(enumerate(st.nnLayers))):
            if l == 0:
                continue  # --- do not check the first hidden layer
            prevLayerActiveNodes = updated_active_nodes[l - 1]
            currLayerActiveNodes = updated_active_nodes[l]
            # ---------------------------------------------------------------------------------------
            # --- Find all ACTIVE nodes of layer 'l-1' that are USED in some node of layer 'l'
            # ---------------------------------------------------------------------------------------
            activeUsedNodes = np.zeros(len(prevLayerActiveNodes), dtype=bool).flatten()
            for n, node in enumerate(layer.units):
                if not currLayerActiveNodes[
                    n
                ]:  # --- only consider active nodes in the current layer
                    continue
                for w in node.weights:
                    if " b" not in w.name:
                        # ---
                        curr_mask = w.mask.numpy().copy().flatten()
                        activeUsedPerNode = (curr_mask == 1) & (
                            np.array(prevLayerActiveNodes)
                        )

                        # log(f"curr_mask: {curr_mask}")
                        # if l == 2 and n == 9:
                        #     wlist = [f"({el1}, {el2})" for el1, el2 in zip(range(len(curr_mask.tolist())), curr_mask.tolist())]
                        #     log(f"curr_mask, node [{l}, {n}]: length={len(wlist)}, {wlist}", level="HIGHLIGHT_CYAN")
                        #     log(f"prevLayerActiveNodes: length={len(prevLayerActiveNodes)}, {prevLayerActiveNodes}")
                        #     log(f"activeUsedPerNode [{l}, {n}]: shape={activeUsedPerNode.shape}, {activeUsedPerNode}", level="HIGHLIGHT_GREEN")

                        activeUsedPerNode = activeUsedPerNode.astype(bool, copy=False)
                        activeUsedNodes |= activeUsedPerNode
            # log(f"activeUsedNodes in layer {l-1}: shape={activeUsedNodes.shape}, {activeUsedNodes}", level="HIGHLIGHT_MAGENTA")
            # --------------------------------------------------------------------------------------------
            # --- Inactivate all active nodes in layer 'l-1' that are not used in any node of layer 'l'
            # --------------------------------------------------------------------------------------------
            for n, currentlyActive in enumerate(prevLayerActiveNodes):
                if currentlyActive and not activeUsedNodes[n]:
                    prevLayerActiveNodes[n] = False
                    node_to_inactivate = st.nnLayers[l - 1].units[n]
                    for w in node_to_inactivate.weights:
                        w.set_allzero_value_mask()
                    # --- reset inheritedFrom
                    if inheritedFrom is not None:
                        inheritedFrom[l - 1][n] = -1

        st.activeNodeCoordinates = updated_active_nodes
        return st, inheritedFrom

    def generate_random_weights_with_max_arity(
        self,
        w: MyTorchVariable,
        max_arity: int,
        layer_id: int,
        inputSize: int,
        completeInputLayerActiveNodes: list[bool],
    ):
        inputLayerActiveNodes: list[bool]
        if " b" in w.name:  # --- bias
            inputLayerActiveNodes = [True]
        elif layer_id == 0:
            # --- First layer: choose a subset of input variables
            # --- 1. random number of active nodes (at least 1)
            k = SRConfig.r.integers(1, min(max_arity, inputSize) + 1)
            # --- 2. choose k unique indices
            active_idx = SRConfig.r.choice(inputSize, size=k, replace=False)
            # --- 3. create boolean list
            inputLayerActiveNodes = [False] * inputSize
            for i in active_idx:
                inputLayerActiveNodes[i] = True
        else:
            # --- All other hidden layers but the last one:
            # --- take active nodes from the subTopologyA, while respecting the maximum arity of the unit
            inputLayerActiveNodes = completeInputLayerActiveNodes.copy()
            true_indices = [i for i, v in enumerate(inputLayerActiveNodes) if v]
            if len(true_indices) > max_arity:
                # --- number of True values to remove
                n_remove = len(true_indices) - max_arity
                # --- randomly choose which True indices to turn off
                remove_idx = SRConfig.r.choice(
                    true_indices, size=n_remove, replace=False
                )
                for i in remove_idx:
                    inputLayerActiveNodes[i] = False
        w.setRandomWeights(
            layerId=layer_id,
            inputLayerActiveNodes=inputLayerActiveNodes,
        )
        return w, inputLayerActiveNodes

    def crossover_limited_arity_topology_mixing(
        self, subTopologyA: SubTopology = None, subTopologyB: SubTopology = None
    ):
        """
        Forward pass!
        Takes into account maximum arity of units!
        Mixes parental topology structures, i.e., mixes active nodes from both parents!
        Inherits activeNodeCoordinates[l][n] either from subTopologyA or from subTopologyB.
        Weights are inherited with a probability SRConfig.inheritWeights.
        Otherwise they are taken from an existing config, if applicable, or randomly regenerated anew.
        :param subTopologyA:
        :param subTopologyB:
        :return: res ... a new subTopology as a combination of the two parents
                 inheritedFrom ... layer-wise information about the origin of each node (1 for subTopologyA, 2 for subTopologyB, -1 if not inherited).
        """
        pExchange = 0.5  # --- probability of inheriting active node from subTopologyA (vs subTopologyB)
        res: SubTopology = (
            subTopologyA.cloneSubTopology()
        )  # --- Default: Use subTopologyA
        inheritedFrom = [
            [-1 for _ in layer.units] for layer in res.nnLayers
        ]  # --- -1 if not inherited, 1 if inherited from A, 2 if inherited from B

        # -----------------------------------------
        # --- Topology structure: Forward pass
        # -----------------------------------------
        offspringActiveNodes = []
        for l, layer in enumerate(res.nnLayers):
            activeNodesLayer = []
            for n, node in enumerate(layer.units):
                inherited: bool = False
                if SRConfig.r.random() < pExchange:
                    currValue = subTopologyA.activeNodeCoordinates[l][n]
                    if (l == len(res.nnLayers) - 1) or (
                        currValue and SRConfig.r.random() < SRConfig.inheritWeights
                    ):
                        # --- INHERIT weights from subTopologyA; weights already set
                        inherited = True
                        inheritedFrom[l][n] = 1
                else:
                    currValue = subTopologyB.activeNodeCoordinates[l][n]
                    if (l == len(res.nnLayers) - 1) or (
                        currValue and SRConfig.r.random() < SRConfig.inheritWeights
                    ):
                        # --- INHERIT weights from subTopologyB; weights must be set accordingly first
                        inherited = True
                        inheritedFrom[l][n] = 2
                        for w, wB in zip(
                            node.weights, subTopologyB.nnLayers[l].units[n].weights
                        ):
                            w.set_value_mask(wB.v)
                            w.adjust_value_mask_for_max_arity()

                if not currValue:  # --- It's an inactive node, set all values to zero
                    for w in node.weights:
                        w.set_allzero_value_mask()
                elif not inherited:  # --- DON'T inherit weights; choose new ones
                    if "UnitIdent1" in node.name:
                        node.reset_weights_and_bias()  # --- UnitIdent1 node: set its default
                    else:
                        for (
                            w
                        ) in node.weights:  # --- Ordinary unit: generate random values
                            w = self.generate_random_weights_with_max_arity(
                                # --- don't consider max arity at this point,
                                #     as it will be taken into account when generating weights for the next layer
                                w,
                                max_arity=sys.maxsize,
                                layer_id=l,
                                inputSize=layer.inSize,
                                completeInputLayerActiveNodes=(
                                    offspringActiveNodes[l - 1] if l > 0 else None
                                ),
                            )
                activeNodesLayer.append(currValue)
            offspringActiveNodes.append(activeNodesLayer)

        # -----------------------------------------
        # --- Weights adjustment: Forward pass
        # -----------------------------------------
        for l, layer in enumerate(res.nnLayers):
            activeNodesLayer = []
            # ---
            for n, node in enumerate(layer.units):
                if SRConfig.r.random() < pExchange:
                    # -----------------------------------------------------------
                    # --- Use subTopologyA
                    # -----------------------------------------------------------
                    currValue = subTopologyA.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    if currValue:
                        if (
                            l < len(res.nnLayers) - 1
                        ) and SRConfig.r.random() > SRConfig.inheritWeights:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    w = self.generate_random_weights_with_max_arity(
                                        w,
                                        max_arity=SRConfig.max_arities.get(
                                            node.name, sys.maxsize
                                        ),
                                        layer_id=l,
                                        inputSize=layer.inSize,
                                        completeInputLayerActiveNodes=(
                                            offspringActiveNodes[l - 1]
                                            if l > 0
                                            else None
                                        ),
                                    )
                        else:
                            # ----------------------------------------
                            # --- Inherit weights from subTopologyA
                            # ----------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    if " b" not in w.name:
                                        w.adjust_value_mask_for_max_arity()
                            inheritedFrom[l][n] = 1
                    else:
                        # ----------------------------------------------------
                        # --- It's an inactive node, set all values to zero
                        # ----------------------------------------------------
                        for w in node.weights:
                            w.set_allzero_value_mask()
                else:
                    # ----------------------------------------------------------
                    # --- Use subTopologyB
                    # ----------------------------------------------------------
                    currValue = subTopologyB.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    if not currValue:
                        # --- It's an inactive node, set all values to zero
                        for w in node.weights:
                            w.set_allzero_value_mask()
                    else:
                        if (
                            l == len(res.nnLayers) - 1
                        ) or SRConfig.r.random() < SRConfig.inheritWeights:
                            # ------------------------------------------------------------------
                            # --- INHERIT weights from subTopologyB,
                            # --- if subTopologyA and subTopologyB have the same active nodes
                            # --- in the previous layer
                            # ------------------------------------------------------------------
                            inheritedFrom[l][n] = 2
                            for wA, wB in zip(
                                node.weights, subTopologyB.nnLayers[l].units[n].weights
                            ):
                                if wA.mask.shape == wB.mask.shape:
                                    wA.set_value_mask(wB.v)
                                    wA.adjust_value_mask_for_max_arity()
                        else:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            if "UnitIdent1" in node.name:
                                # --- UnitIdent1 node: set its default
                                node.reset_weights_and_bias()
                            else:
                                # --- Ordinary unit: generate random values
                                for w in node.weights:
                                    w = self.generate_random_weights_with_max_arity(
                                        w,
                                        max_arity=SRConfig.max_arities.get(
                                            node.name, sys.maxsize
                                        ),
                                        layer_id=l,
                                        inputSize=layer.inSize,
                                        completeInputLayerActiveNodes=(
                                            offspringActiveNodes[l - 1]
                                            if l > 0
                                            else None
                                        ),
                                    )
            offspringActiveNodes.append(activeNodesLayer)
        res.activeNodeCoordinates = offspringActiveNodes
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(res.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(res, l, n)
        # ---
        return res, inheritedFrom

    def networkActivityMutation(self, ind: SubTopology):
        """
        Mutate all activeNodeCoordinates elements with certain probability.
        :return:
        """
        changed: bool = False
        changedNodes: list[list[bool]] = [
            [False for _ in range(len(layer.units))] for layer in ind.nnLayers
        ]
        for l in range(len(ind.activeNodeCoordinates) - 1):  # --- ORIGINAL
            for n in range(len(ind.activeNodeCoordinates[l])):
                value = ind.activeNodeCoordinates[l][n]
                random = SRConfig.r.random()
                if (
                    (
                        self.nodeUtilizations.nodeUtilizationsRel[l][n]
                        <= SRConfig.utilLow
                        and not value
                    )
                    or (
                        self.nodeUtilizations.nodeUtilizationsRel[l][n]
                        >= SRConfig.utilHigh
                        and value
                    )
                ) and (random < SRConfig.nodeActivityMutationRate):
                    changedNodes[l][n] = True
                    changed = True
                    ind.activeNodeCoordinates[l][n] = not value
                    if ind.activeNodeCoordinates[l][n]:
                        # --- node became active
                        if "UnitIdent1" in ind.nnLayers[l].units[n].name:
                            # --------------------------------------------
                            # --- UnitIdent1 node: set its default
                            # --------------------------------------------
                            ind.nnLayers[l].units[n].reset_weights_and_bias()
                        else:
                            # --------------------------------------------
                            # --- Ordinary unit: generate random values
                            # --------------------------------------------
                            for w in ind.nnLayers[l].units[n].weights:
                                inputLayerActiveNodes: list[bool]
                                if " b" in w.name:  # --- bias
                                    inputLayerActiveNodes = [True]
                                elif l == 0:
                                    inputLayerActiveNodes = ind.nnLayers[l].inSize * [
                                        True
                                    ]
                                else:
                                    inputLayerActiveNodes = ind.activeNodeCoordinates[
                                        l - 1
                                    ]
                                if sum(inputLayerActiveNodes) > 0:
                                    w.setRandomWeights(
                                        layerId=l,
                                        inputLayerActiveNodes=inputLayerActiveNodes,
                                    )
                    else:
                        # --- node became inactive
                        for w in ind.nnLayers[l].units[n].weights:
                            w.set_allzero_value_mask()
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(ind.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(ind, l, n)
        # ---
        return changed, changedNodes

    def networkActivityMutationOnlyToInactive(self, ind: SubTopology):
        """
        Mutate all activeNodeCoordinates elements with certain probability. Allow only mutation from active to inactive.
        :return:
        """
        changed: bool = False
        for l in range(len(ind.activeNodeCoordinates) - 1):  # --- ORIGINAL
            for n in range(len(ind.activeNodeCoordinates[l])):
                value = ind.activeNodeCoordinates[l][n]
                if not value:
                    continue  # --- do not activate inactive nodes
                random = SRConfig.r.random()
                if (
                    (
                        self.nodeUtilizations.nodeUtilizationsRel[l][n]
                        >= SRConfig.utilHigh
                        and value
                    )
                ) and (random < SRConfig.nodeActivityMutationRate):
                    changed = True
                    ind.activeNodeCoordinates[l][n] = not value
                    # --- node became inactive
                    for w in ind.nnLayers[l].units[n].weights:
                        w.set_allzero_value_mask()
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(ind.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(ind, l, n)
        # ---
        return changed

    def networkConfigMutation_A(self, ind: SubTopology):
        """
        Forward pass!
        Mutates active nodes weights.
        :param ind:
        :return:
        """
        for l in range(len(ind.activeNodeCoordinates)):
            for n in range(len(ind.activeNodeCoordinates[l])):
                if ind.activeNodeCoordinates[l][n] and (
                    SRConfig.r.random() < SRConfig.nodeConfigMutationRate
                ):
                    # --- choose randomly one config
                    for w in ind.nnLayers[l].units[n].weights:
                        # --- generate random values
                        inputLayerActiveNodes: list[bool]
                        if " b" in w.name:  # --- bias
                            inputLayerActiveNodes = [True]
                        elif l == 0:
                            inputLayerActiveNodes = ind.nnLayers[l].inSize * [True]
                        else:
                            inputLayerActiveNodes = ind.activeNodeCoordinates[l - 1]
                        w.setRandomWeights(
                            layerId=l, inputLayerActiveNodes=inputLayerActiveNodes
                        )

    def chooseLayerProportionallyNodeUniformly(self, ind: SubTopology):
        """
        Chooses randomly one layer proportionally to its depth and
        chooses one of its active nodes randomly uniformly.
        :param ind:
        :return:
        """
        x = 1.0 / ((1 + len(self.nnLayers)) * len(self.nnLayers) / 2.0)
        attempts = 10
        layerId = 0
        while True:
            r = SRConfig.r.random()
            n = len(self.nnLayers)
            acc = n * x
            while r > acc and n > 0:
                n -= 1
                acc += n * x
                layerId = len(self.nnLayers) - n
            if np.sum(ind.activeNodeCoordinates[layerId]) > 0:
                break
            attempts -= 1
            if attempts <= 0:
                layerId = None
                break
        if layerId is None:
            return None, None
        while True:
            nodeId = SRConfig.r.integers(low=0, high=len(self.nnLayers[layerId].units))
            if ind.activeNodeCoordinates[layerId][nodeId]:
                break
        return layerId, nodeId

    def chooseLayerUniformlyNodeUniformly(self, ind: SubTopology):
        """
        Chooses randomly one layer uniformly to its depth and
        chooses one of its active nodes randomly uniformly.
        :param ind:
        :return:
        """
        while True:
            layerId = SRConfig.r.integers(low=0, high=len(self.nnLayers))
            if np.sum(ind.activeNodeCoordinates[layerId]) > 0:
                break
        while True:
            nodeId = SRConfig.r.integers(low=0, high=len(self.nnLayers[layerId].units))
            if ind.activeNodeCoordinates[layerId][nodeId]:
                break
        return layerId, nodeId

    def networkConfigMutation_B(self, ind: SubTopology):
        """
        Mutates only ACTIVE NODEs weights.
        Firstly, one layer is randomly chosen using a distribution that linearly increases from the output
        layer to the first one. For L layers, a probability of choosing the last layer is x %, while the probability
        of choosing the first layer if L*x %.
        :param ind:
        :return:
        """
        # -------------------------------------------------------------
        # --- Choose layer and one of its nodes to be selected
        # -------------------------------------------------------------
        layerId, nodeId = self.chooseLayerProportionallyNodeUniformly(ind)
        if layerId is None or nodeId is None:
            return False, None, None
        # -------------------------------------------------------------
        # --- Apply mutation to the selected node
        # -------------------------------------------------------------
        # --- Choose randomly one config, if there is one
        if "UnitIdent1" in ind.nnLayers[layerId].units[nodeId].name:
            # --------------------------------------------
            # --- UnitIdent1 node: set its default
            # --------------------------------------------
            ind.nnLayers[layerId].units[nodeId].reset_weights_and_bias()
        else:
            # --------------------------------------------
            # --- Ordinary unit: generate random values
            # --------------------------------------------
            for w in ind.nnLayers[layerId].units[nodeId].weights:
                inputLayerActiveNodes: list[bool]
                if " b" in w.name:  # --- bias
                    inputLayerActiveNodes = [True]
                else:
                    if layerId == 0:
                        inputLayerActiveNodes = [
                            SRConfig.r.random() < 0.5
                            for _ in range(ind.nnLayers[layerId].inSize)
                        ]
                        if sum(inputLayerActiveNodes) == 0:
                            nonzero: int = SRConfig.r.integers(
                                low=0, high=ind.nnLayers[layerId].inSize
                            )
                            inputLayerActiveNodes[nonzero] = True
                    else:
                        inputLayerActiveNodes = ind.activeNodeCoordinates[layerId - 1]
                w.setRandomWeights(
                    layerId=layerId, inputLayerActiveNodes=inputLayerActiveNodes
                )
        # ---
        return True, layerId, nodeId

    def activateNode(self, subTopology: SubTopology, layerId: int, nodeId: int):
        """
        Activates a node at the given position and makes sure that its weights and mask are initialized.
        :param layerId:
        :param nodeId:
        :return:
        """
        subTopology.activeNodeCoordinates[layerId][nodeId] = True
        if "UnitIdent1" in subTopology.nnLayers[layerId].units[nodeId].name:
            subTopology.nnLayers[layerId].units[nodeId].reset_weights_and_bias()
        else:
            # --- generate random values
            for w in subTopology.nnLayers[layerId].units[nodeId].weights:
                inputLayerActiveNodes: list[bool]
                if " b" in w.name:  # --- bias
                    inputLayerActiveNodes = [True]
                elif layerId == 0:
                    inputLayerActiveNodes = subTopology.nnLayers[0].inSize * [True]
                else:
                    inputLayerActiveNodes = subTopology.activeNodeCoordinates[
                        layerId - 1
                    ]
                w.setRandomWeights(
                    layerId=layerId, inputLayerActiveNodes=inputLayerActiveNodes
                )

    def useActiveNode(self, subTopology: SubTopology, layerId: int, nodeId: int):
        """
        Link the output of the given active node 'nnLayers[layerId].units[nodeId]' with a randomly chosen active node in the subsequent layer 'layerId+1'.
        :param layerId:
        :param nodeId:
        :return:
        """
        # --- Collect ordinary active nodes in the subsequent layer 'l+1'
        activeOrdinaryOutputNodes = [
            i
            for i in range(len(subTopology.activeNodeCoordinates[layerId + 1]))
            if (
                subTopology.activeNodeCoordinates[layerId + 1][i]
                and (
                    "UnitIdent1" not in subTopology.nnLayers[layerId + 1].units[i].name
                )
            )
        ]
        # --------------------------------------------------------------
        # --- Ordinary function unit is to be used in the layer l+1
        # --------------------------------------------------------------
        if "UnitIdent1" not in subTopology.nnLayers[layerId].units[nodeId].name:
            # --- Ordinary unit in layer 'l' can lead only to an ordinary unit in layer 'l+1'
            # --- TODO: ordinary unit can lead to unitIdent1 as well
            if activeOrdinaryOutputNodes:
                res: int = SRConfig.r.choice(
                    activeOrdinaryOutputNodes
                )  # --- choose one unit in the layer 'l+1' that will use the unit nnLayers[layerId].units[nodeId]
                for w in subTopology.nnLayers[layerId + 1].units[res].weights:
                    if " b" not in w.name:  # --- weights, NOT bias
                        values = w.v.clone()
                        # values[nodeId] = SRConfig.r.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=1)     # --- ORIGINAL: uniform
                        if SRConfig.r.random() < 0.5:
                            if (layerId == 0) and (
                                SRConfig.r.random() < SRConfig.weight_init_one
                            ):
                                with torch.no_grad():
                                    values[nodeId] = 1.0  # --- TODO   +1
                            else:
                                with torch.no_grad():
                                    values[nodeId] = SRConfig.weight_min_init + abs(
                                        SRConfig.r.normal(
                                            loc=0.0, scale=SRConfig.weight_sigma, size=1
                                        )[0]
                                    )  # --- TODO: normal with SRConfig.weight_min_init
                        else:
                            if (layerId == 0) and (
                                SRConfig.r.random() < SRConfig.weight_init_one
                            ):
                                with torch.no_grad():
                                    values[nodeId] = -1.0  # --- TODO   -1
                            else:
                                with torch.no_grad():
                                    values[nodeId] = -SRConfig.weight_min_init - abs(
                                        SRConfig.r.normal(
                                            loc=0.0, scale=SRConfig.weight_sigma, size=1
                                        )[0]
                                    )  # --- TODO: normal with SRConfig.weight_min_init
                        w.set_value_mask(values)
                        break  # --- TODO: only one variable is updated
        # --------------------------------------------------------------------------------------
        # --- UnitIdent1 is the unit to be used as an iput to some unit in the next layer l+1
        # --------------------------------------------------------------------------------------
        else:
            # --- Find a corresponding UnitIdent1 id in the subsequent layer layerId+1
            identId: int = 0
            for i, node in enumerate(subTopology.nnLayers[layerId + 1].units):
                node: UnitIdent1
                if ("UnitIdent1" in node.name) and (node.inId == nodeId):
                    identId = i
                    activeOrdinaryOutputNodes.append(
                        i
                    )  # --- this UnitIdent1 node, nnLayers[layerId+1].units[identId], is added to the set of possible consumers of the current node's output
                    break
            # --- Choose randomly one of the output nodes.
            # --- It can be either an ordinary function unit or the particular UnitIdent1.
            if not activeOrdinaryOutputNodes:
                return
            res: int = SRConfig.r.choice(activeOrdinaryOutputNodes)
            if res == identId:
                # --- Output node is UnitIdent1
                subTopology.nnLayers[layerId + 1].units[res].reset_weights_and_bias()
                subTopology.activeNodeCoordinates[layerId + 1][res] = True
            else:
                # --- Output node is an ordinary unit
                for w in subTopology.nnLayers[layerId + 1].units[res].weights:
                    if " b" not in w.name:  # --- weights, NOT bias
                        values = w.v
                        if SRConfig.r.random() < 0.5:
                            # if ((layerId == 0) or ('x[' in subTopology.nnLayers[layerId].units[nodeId].proxyName)) and (SRConfig.r.random() < SRConfig.weight_init_one):
                            if (layerId == 0) and (
                                SRConfig.r.random() < SRConfig.weight_init_one
                            ):
                                with torch.no_grad():
                                    values[nodeId] = 1.0  # --- TODO   +1
                            else:
                                with torch.no_grad():
                                    values[nodeId] = SRConfig.weight_min_init + abs(
                                        SRConfig.r.normal(
                                            loc=0.0, scale=SRConfig.weight_sigma, size=1
                                        )[0]
                                    )  # --- TODO: normal with SRConfig.weight_min_init
                        else:
                            # if ((layerId == 0) or ('x[' in subTopology.nnLayers[layerId].units[nodeId].proxyName)) and (SRConfig.r.random() < SRConfig.weight_init_one):
                            if (layerId == 0) and (
                                SRConfig.r.random() < SRConfig.weight_init_one
                            ):
                                with torch.no_grad():
                                    values[nodeId] = -1.0  # --- TODO   -1
                            else:
                                with torch.no_grad():
                                    values[nodeId] = -SRConfig.weight_min_init - abs(
                                        SRConfig.r.normal(
                                            loc=0.0, scale=SRConfig.weight_sigma, size=1
                                        )[0]
                                    )  # --- TODO: normal with SRConfig.weight_min_init
                        w.set_value_mask(values)

    def rectifyActiveNodesConnectivity(self, subTopology: SubTopology):
        """
        Forward pass only.
        For each layer, from the output one forward, apply the following actions to all its active nodes:
           1) Make sure that the node weights has some weight config chosen.
                    If the active node is the UnitIdent1 then set its mask to mask_orig and set its weights accordingly.
           2) Make sure that all its active weights point to active nodes. If some active weight links to an inactive node then
                  - the weight is inactivated,
                  - the mask is adjusted accordingly.
               If the weight pointing to an inactive node is the last active one then
                  - another inactive weight pointing to an active node is activated,
                  - the mask is adjusted accordingly.
           3) Make sure that the node's output is passed to at least one node in the subsequent layer.
              The output node is chosen only among the set of active nodes of the subsequent layer.
              If not applicable, do nothing.
        :return:
        """
        if SRConfig.r.random() > SRConfig.rectify:  # --- do not perform rectification
            return
        # ---
        for l in range(len(subTopology.nnLayers)):  # --- l ... current layer
            # ----------------------------------------
            # --- All layers, but the first one.
            # ----------------------------------------
            if l > 0:
                # --- input-nodes activity accumulation
                for n, currentNode in enumerate(subTopology.nnLayers[l].units):
                    for tfvar in currentNode.weights:
                        if " b" not in tfvar.name:  # --- weights, NOT bias
                            if "UnitIdent1" in currentNode.name:
                                # ----------------------------------------------------------------------------
                                # --- 1) If the active node is the UnitIdent1
                                # ---    then set its mask to mask_orig and set its weights accordingly.
                                # ----------------------------------------------------------------------------
                                currentNode: UnitIdent1
                                if not subTopology.activeNodeCoordinates[l - 1][
                                    currentNode.inId
                                ]:
                                    # --- Unsatisfiable: deactivate UnitIdent1
                                    subTopology.activeNodeCoordinates[l][n] = False
                                    tfvar.set_allzero_value_mask()
                            else:
                                # ----------------------------------------------------------------------------
                                # --- 2) Make sure that all its active weights point to active nodes.
                                # ----------------------------------------------------------------------------
                                unsatisfiedActivity = [
                                    req and (not act)
                                    for req, act in zip(
                                        tfvar.mask.flatten(),
                                        subTopology.activeNodeCoordinates[l - 1],
                                    )
                                ]
                                if sum(unsatisfiedActivity) > 0:
                                    weights = tfvar.v
                                    for i, u in enumerate(unsatisfiedActivity):
                                        if u:
                                            with torch.no_grad():
                                                weights[i] = 0
                                    tfvar.set_value_mask(weights)
                                    # ---
                                    if sum(weights) == 0:
                                        origMask = np.array(
                                            subTopology.activeNodeCoordinates[l - 1]
                                        ).reshape(
                                            (
                                                len(
                                                    subTopology.activeNodeCoordinates[
                                                        l - 1
                                                    ]
                                                ),
                                                1,
                                            )
                                        )
                                        origMaskNonZero = origMask.nonzero()
                                        if len(origMaskNonZero[0]) > 0:
                                            toActivate = SRConfig.r.choice(
                                                origMaskNonZero[0], 1
                                            )[0]
                                            # weights[toActivate] = SRConfig.r.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=1)    # --- ORIGINAL: uniform
                                            if SRConfig.r.random() < 0.5:
                                                with torch.no_grad():
                                                    weights[toActivate] = (
                                                        SRConfig.weight_min_init
                                                        + abs(
                                                            SRConfig.r.normal(
                                                                loc=0.0,
                                                                scale=SRConfig.weight_sigma,
                                                                size=1,
                                                            )[0]
                                                        )
                                                    )  # --- TODO: normal with weight_min_init
                                            else:
                                                with torch.no_grad():
                                                    weights[toActivate] = (
                                                        -SRConfig.weight_min_init
                                                        - abs(
                                                            SRConfig.r.normal(
                                                                loc=0.0,
                                                                scale=SRConfig.weight_sigma,
                                                                size=1,
                                                            )[0]
                                                        )
                                                    )  # --- TODO: normal with weight_min_init
                                            tfvar.set_value_mask(weights)
                                        else:
                                            # --- Unsatisfiable: deactivate node[l][n]
                                            subTopology.activeNodeCoordinates[l][
                                                n
                                            ] = False
            # ----------------------------------------
            # --- All layers, but the last one.
            # ----------------------------------------
            if l < len(subTopology.nnLayers) - 1:
                # --- 3) Make sure that the node's output is passed to at least one node in the subsequent layer.
                outputNodesInputActivity = np.zeros(
                    shape=[len(subTopology.nnLayers[l].units), 1]
                )  # --- aggregated input activity over all active nodes in the next layer 'l+1'
                # --- input-nodes activity aggregation
                for outputNode in subTopology.nnLayers[l + 1].units:
                    for tfvar in outputNode.weights:
                        if " b" not in tfvar.name:  # --- weights, NOT bias
                            outputNodesInputActivity = np.logical_or(
                                outputNodesInputActivity, tfvar.mask
                            )
                # --- output layer activity check
                for i, u in enumerate(subTopology.activeNodeCoordinates[l]):
                    if (
                        u and not outputNodesInputActivity[i]
                    ):  # --- the active node 'nnLayers[l].units[i]' IS NOT used by any active node in the subsequent layer 'l+1'
                        self.useActiveNode(subTopology, l, i)
