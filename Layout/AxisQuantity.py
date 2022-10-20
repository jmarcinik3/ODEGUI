from __future__ import annotations
import traceback
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import PySimpleGUI as sg
from Config import loadConfig, saveConfig
from macros import getIndicies, recursiveMethod, removeAtIndicies

from Layout.Layout import (CheckboxGroup, Frame, Layout, RadioGroup, Row, Tab,
                           TabGroup, getKeys, storeElement)

if TYPE_CHECKING:
    from Layout.SimulationWindow import (SimulationWindow,
                                         SimulationWindowRunner)

cc_pre = "CANVAS CHOICE"
ccs_pre = ' '.join((cc_pre, "SPECIE"))
fps_pre = "FREE_PARAMETER SLIDER"
qr_pre = "QUANTITY RESET"

axis_type2names = {
    "contour": ('C', ),
    "grid": ('X', 'Y'),
    "color": ('c',),
    "cartesian": ('x', 'y', 'z')
}


def getAxisNames(
    filter: str = None,
    axis_type2names: Dict[str, Tuple[str, ...]] = axis_type2names
) -> List[str]:
    """
    Get names for axes.

    :param axis_type2names: dictionary of axis names.
        Key (str) is type of axis.
        Value is name of axes of type.
    :param filter: only include axis types that satisfy this filter (e.g. "cartesian", "grid", or "color")
    :returns: List of axis names (e.g. ['x', 'y', 'c'])
    """
    if filter is None:
        all_names = [
            axis_name
            for axis_names in axis_type2names.values()
            for axis_name in axis_names
        ]
        return all_names
    elif isinstance(filter, str):
        filter = filter.lower()
        return axis_type2names[filter]
    else:
        raise TypeError("filter must be str")


def generateMetadataFromFile(filepath: str) -> AxisQuantityMetadata:
    """
    Generate axis-quantity object from config file.

    :param self: :class:`~layout.AxisQuantity.AxisQuantityMetadata` to load into
    :param filepath: path of file to load from
    """
    axis_quantity_metadata = loadConfig(filepath)
    axis_name = axis_quantity_metadata["axis_name"]
    axis_quantity_metadata_obj = AxisQuantityMetadata(axis_name)
    axis_quantity_metadata_obj.loadMetadataFromFile(filepath)
    return axis_quantity_metadata_obj


class AxisQuantityMetadata:
    def __init__(
        self,
        axis_name: str
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityMetadata`.

        :param axis_name: name of axis to organize metadata for
        """
        self.axis_name = axis_name
        self.reset()

    def reset(self) -> None:
        """
        Reset all attributes of class by looking at GUI elements (i.e. set all attributes to NoneType)

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to reset attributes in
        """
        self.specie_names: List[str] = None
        self.quantity_names: List[str] = None
        self.envelope_name: str = None
        self.functional_name: str = None
        self.transform_name: str = None
        self.complex_name: str = None
        self.normalize_names: List[str] = None
        self.functional_names: List[str] = None
        self.parameter_namess: List[List[str]] = None
        self.quantity_type: str = None

    def getAxisName(self) -> str:
        """
        Get name of corresponding axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        """
        return self.axis_name

    def getSpecieNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected specie names for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        specie_names = self.specie_names
        if not include_none:
            none_indicies = getIndicies("None", specie_names)
            specie_names = removeAtIndicies(specie_names, none_indicies)

        return specie_names

    def getAxisQuantitySpecie(
        self,
        index: int
    ) -> str:
        """
        Get selected quantity specie for desired axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve quantity name from
        :param index: index of axis quantity per axis name
        """
        specie_names = self.getSpecieNames()
        specie_name = specie_names[index]
        return specie_name

    def existsLikeSpecies(self, like: str) -> bool:
        """
        Get whether or not axis-quantity contains at least one species in like-type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve specie names from
        :param like: species type to determine whether axis-quantity contains at least one in
        """
        specie_names = self.getSpecieNames()
        exists_like = PlotQuantities.existsLikeSpecies(specie_names, like)
        return exists_like

    def getLikeCount(self, like: str) -> int:
        """
        Get number of species for axis-quantity in like-type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve specie names from
        :param like: species type to count number of species in
        """
        specie_names = self.getSpecieNames()
        like_count = PlotQuantities.getLikeCount(specie_names, like)
        return like_count

    def existsNontimelikeSpecies(self) -> bool:
        """
        Get whether axis has at least one nontimelike species.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        :returns: True if axis contains at least one nontimelike species.
            False if all species are timelike.
        """
        specie_names = self.getSpecieNames()
        getSpecies = PlotQuantities.getSpecies
        timelike_species = getSpecies("timelike") + getSpecies("nonelike")

        exists_nontimelike_specie = False
        for specie_name in specie_names:
            if specie_name not in timelike_species:
                exists_nontimelike_specie = True
                break

        return exists_nontimelike_specie

    def getAxisQuantityNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected quantity names for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        quantity_names = self.quantity_names
        if not include_none:
            specie_names = self.getSpecieNames(include_none=True)
            none_indicies = getIndicies("None", specie_names)
            quantity_names = removeAtIndicies(quantity_names, none_indicies)

        return quantity_names

    def getAxisQuantityName(
        self,
        index: int
    ) -> str:
        """
        Get selected quantity name for subaxis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        :param index: index of axis quantity per axis name
        """
        quantity_names = self.getAxisQuantityNames()
        quantity_name = quantity_names[index]
        return quantity_name

    def getEnvelopeName(self) -> str:
        """
        Get name of envelope (e.g. "Amplitude") for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        """
        return self.envelope_name

    def getFunctionalName(self) -> str:
        """
        Get name of functional for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        """
        return self.functional_name

    def isFunctional(self) -> bool:
        """
        Get whether or not axis quantity is condensed by some functional function.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        functional_name = self.getFunctionalName()
        is_functional = functional_name != "None"
        return is_functional

    def getTransformName(self) -> str:
        """
        Get name of transform for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        """
        return self.transform_name

    def isTransformed(self) -> bool:
        """
        Get whether or not axis quantity is transformed by some transform function.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        transform_name = self.getTransformName()
        is_transformed = transform_name != "None"
        return is_transformed

    def getComplexName(self) -> str:
        """
        Get name of complex-reduction method for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        """
        return self.complex_name

    def getNormalizeNames(self) -> List[str]:
        """
        Get name of parameters to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve names from
        """
        return self.normalize_names

    def getFunctionalFunctionalNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get names of functionals for axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve name from
        :param include_none: set True to include "None" functional. Set False otherwise.
        """
        functional_names = self.functional_names
        if not include_none:
            none_indicies = getIndicies("None", functional_names)
            functional_names = removeAtIndicies(functional_names.copy(), none_indicies)

        return functional_names

    def getFunctionalParameterNamess(
        self,
        include_none: bool = True
    ) -> List[List[str]]:
        """
        Get name of axes to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve names from
        :param include_none: set True to include "None" functional. Set False otherwise.
        """
        parameter_namess = self.parameter_namess
        if not include_none:
            functional_names = self.getFunctionalFunctionalNames(include_none=True)
            none_indicies = getIndicies("None", functional_names)
            parameter_namess = removeAtIndicies(parameter_namess.copy(), none_indicies)

        return parameter_namess

    def getQuantityType(self) -> str:
        """
        Get type of quantity corresponding to axis-quantity.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve type from
        :returns: "nonelike", "timelike", "parameterlike", or "functionallike"
        """
        return self.quantity_type

    def isTimeLike(self) -> bool:
        """
        Get whether axis-quantity is of "timelike" type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        quantity_type = self.getQuantityType()
        is_timelike = quantity_type == "timelike"
        return is_timelike

    def isFunctionalLike(self) -> bool:
        """
        Get whether axis-quantity is of "functionallike" type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        quantity_type = self.getQuantityType()
        is_functionallike = quantity_type == "functionallike"
        return is_functionallike

    def isParameterLike(self) -> bool:
        """
        Get whether axis-quantity is of "parameterlike" type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        quantity_type = self.getQuantityType()
        is_parameterlike = quantity_type == "parameterlike"
        return is_parameterlike

    def isNoneLike(self) -> bool:
        """
        Get whether axis-quantity is of "nonelike" type.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve boolean from
        """
        quantity_type = self.getQuantityType()
        is_nonelike = quantity_type == "nonelike"
        return is_nonelike

    def loadMetadataFromFile(self, filepath: str) -> None:
        """
        Load axis-quantity attributes from config file.

        :param self: :class:`~layout.AxisQuantity.AxisQuantityMetadata` to load into
        :param filepath: path of file to load from
        """
        assert isinstance(filepath, str)

        axis_quantity_metadata = loadConfig(filepath)

        transformations_metadata = axis_quantity_metadata["transformations"]
        for transformation_metadata in transformations_metadata:
            transformation_type = transformation_metadata["type"]
            if transformation_type == "envelope":
                envelope_name = transformation_metadata["name"]
                self.envelope_name = envelope_name
            elif transformation_type == "transform":
                transform_name = transformation_metadata["name"]
                self.transform_name = transform_name
            elif transformation_type == "temporal_functional":
                temporal_functional_name = transformation_metadata["name"]
                self.functional_name = temporal_functional_name
            elif transformation_type == "parameter_functional":
                parameter_functional_name = transformation_metadata["name"]
                parameter_names = transformation_metadata["parameter_names"]
                self.functional_names = [parameter_functional_name]
                self.parameter_namess = [parameter_names]
            elif transformation_type == "scale_factor":
                scale_factor = transformation_metadata["value"]

        quantities_metadata = axis_quantity_metadata["quantities"]
        specie_names = []
        quantity_names = []
        for quantity_metadata in quantities_metadata:
            specie_name = quantity_metadata["specie_name"]
            specie_names.append(specie_name)
            quantity_name = quantity_metadata["quantity_name"]
            quantity_names.append(quantity_name)
        self.specie_names = specie_names
        self.quantity_names = quantity_names

    def saveMetadataToFile(
        self,
        filepath: str,
        fit_parameter_names: Optional[List[str]] = None
    ) -> None:
        """
        Save metadata for axis-quantity into config file.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityMetadata` to retrieve metadata from
        :param filepath: path of file to save metadata into
        :param fit_parameter_names: names of fit parameters to add into metadata (optional)
        """
        assert isinstance(filepath, str)

        self.reset()
        axis_quantity_metadata = {}
        axis_quantity_metadata["axis_name"] = self.getAxisName()

        envelope_name = self.getEnvelopeName()
        envelope_transformation = {
            "type": "envelope",
            "name": envelope_name
        }
        transform_name = self.getTransformName()
        transform_transformation = {
            "type": "transform",
            "name": transform_name
        }

        temporal_functional_name = self.getFunctionalName()
        temporal_functional_transformation = {
            "type": "temporal_functional",
            "name": temporal_functional_name
        }

        parameter_functional_names = self.getFunctionalFunctionalNames()
        functional_parameter_namess = self.getFunctionalParameterNamess()
        parameter_functional_transformations = []
        for parameter_functional_name, functional_parameter_names in zip(parameter_functional_names, functional_parameter_namess):
            parameter_functional_transformation = {
                "type": "parameter_functional",
                "name": parameter_functional_name,
                "parameter_names": functional_parameter_names
            }
            parameter_functional_transformations.append(parameter_functional_transformation)

        transformations = [
            envelope_transformation,
            transform_transformation,
            temporal_functional_transformation,
            *parameter_functional_transformations
        ]
        axis_quantity_metadata["transformations"] = transformations

        specie_names = self.getSpecieNames()
        quantity_names = self.getAxisQuantityNames()
        quantities = []

        for specie_name, quantity_name in zip(specie_names, quantity_names):
            quantity = {
                "specie_name": specie_name,
                "quantity_name": quantity_name
            }
            quantities.append(quantity)
        axis_quantity_metadata["quantities"] = quantities

        if fit_parameter_names is not None:
            assert isinstance(fit_parameter_names)
            for fit_parameter_name in fit_parameter_names:
                assert isinstance(fit_parameter_name, str)
            axis_quantity_metadata["fit_parameter_names"] = fit_parameter_names

        saveConfig(axis_quantity_metadata, filepath)


class AxisQuantity(AxisQuantityMetadata):
    def __init__(
        self,
        axis_quantity_frame: AxisQuantityFrame
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantity`.

        :param axis_quantity_frame: frame to retrieve metadata from
        """
        assert isinstance(axis_quantity_frame, AxisQuantityFrame)
        self.axis_quantity_frame = axis_quantity_frame
        axis_name = axis_quantity_frame.getName()

        AxisQuantityMetadata.__init__(
            self,
            axis_name=axis_name
        )

    def updateAttributes(self) -> None:
        """
        Update all attributes of class by looking at GUI elements.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantity` to update attributes in
        """
        self.reset()
        self.generateSpecieNames()
        self.generateQuantityNames()
        self.generateEnvelopeName()
        self.generateFunctionalName()
        self.generateTransformName()
        self.generateComplexName()
        self.generateNormalizeNames()
        self.generateFunctionalFunctionalNames()
        self.generateFunctionalParameterNamess()

    def updatePlotChoices(self) -> None:
        """
        Update choices for quantities in combobox.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantity` to update choices in
        """
        getSpecies = PlotQuantities.getSpecies
        timelike_species = getSpecies("timelike") + getSpecies("nonelike")
        all_species = getSpecies()
        nontimelike_species = [
            specie
            for specie in all_species
            if specie not in timelike_species
        ]

        axis_quantity_frame = self.getAxisQuantityFrame()
        window_runner: SimulationWindowRunner = axis_quantity_frame.getWindowRunner()

        species = self.generateSpecieNames()
        axis_quantity_rows: List[AxisQuantityRow] = axis_quantity_frame.getAxisQuantityRows()
        for index, axis_quantity_row in enumerate(axis_quantity_rows):
            specie = species[index]
            quantity_names_combobox = axis_quantity_row.getAxisQuantityElement()

            new_choices = window_runner.getPlotChoices(species=specie) if specie != "None" else ['']
            old_choices = vars(quantity_names_combobox)["Values"]
            if new_choices != old_choices:
                quantity_names_combobox_key = getKeys(quantity_names_combobox)

                kwargs = {
                    "values": new_choices,
                    "disabled": specie == "None",
                    "size": vars(quantity_names_combobox)["Size"]
                }

                old_chosen_quantity = window_runner.getValue(quantity_names_combobox_key)
                change_quantity = old_chosen_quantity not in new_choices
                if change_quantity:
                    new_chosen_quantity = new_choices[0]
                    kwargs["value"] = new_chosen_quantity
                    quantity_names_combobox.update(**kwargs)
                    window = window_runner.getWindow()
                    window.write_event_value(quantity_names_combobox_key, new_chosen_quantity)
                else:
                    quantity_names_combobox.update(**kwargs)

        exists_nontimelike = False
        for nontimelike_specie in nontimelike_species:
            if nontimelike_specie in species:
                exists_nontimelike = True
                break
        envelope_disabled = transform_disabled = functional_disabled = complex_disabled = exists_nontimelike

        envelope_radio_group = axis_quantity_frame.getAxisEnvelopeGroup()
        if isinstance(envelope_radio_group, EnvelopeRadioGroup):
            envelope_radios = envelope_radio_group.getRadios()
            for envelope_radio in envelope_radios:
                envelope_radio.update(disabled=envelope_disabled)

        transform_combobox = axis_quantity_frame.getAxisTransformElement()
        if isinstance(transform_combobox, sg.InputCombo):
            transform_combobox.update(disabled=transform_disabled)

        functional_combobox = axis_quantity_frame.getAxisFunctionalElement()
        if isinstance(functional_combobox, sg.InputCombo):
            functional_combobox.update(disabled=functional_disabled)

        complex_radio_group = axis_quantity_frame.getAxisComplexGroup()
        if isinstance(complex_radio_group, ComplexRadioGroup):
            complex_radios = complex_radio_group.getRadios()
            for complex_radio in complex_radios:
                complex_radio.update(disabled=complex_disabled)

        self.reset()

    def getAxisQuantityFrame(self) -> AxisQuantityFrame:
        """
        Get axis-quantity frame to retrieve axis-quantity properties from.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantity` to retrieve frame from
        """
        return self.axis_quantity_frame

    def generateMetadata(self) -> None:
        """
        Generate all metadata attributes for axis-quantity, from axis-quantity frame.

        :param self: class:`~Layout.AxisQuantity.AxisQuantity` to generate metadata in
        """
        self.generateSpecieNames()
        self.generateQuantityNames()
        self.generateEnvelopeName()
        self.generateFunctionalName()
        self.generateTransformName()
        self.generateComplexName()
        self.generateNormalizeNames()
        self.generateFunctionalFunctionalNames()
        self.generateFunctionalParameterNamess()
        self.generateQuantityType()

    def generateSpecieNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected specie names for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        specie_names = self.getSpecieNames(include_none=include_none)
        if specie_names is None:
            axis_quantity_frame = self.getAxisQuantityFrame()
            quantity_count_per_axis = axis_quantity_frame.getQuantityCountPerAxis()

            specie_names = []
            window_runner = axis_quantity_frame.getWindowRunner()
            for index in range(quantity_count_per_axis):
                axis_quantity_row = axis_quantity_frame.getAxisQuantityRows(indicies=index)
                specie_element = axis_quantity_row.getAxisQuantitySpeciesElement()
                specie_name_key = getKeys(specie_element)
                specie_name = window_runner.getValue(specie_name_key)
                specie_names.append(specie_name)

            self.specie_names = specie_names.copy()
            specie_names = self.getSpecieNames(include_none=include_none)

        return specie_names

    def generateAxisQuantitySpecie(
        self,
        index: int
    ) -> str:
        """
        Get selected quantity specie for desired axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve quantity name from
        :param index: index of axis quantity per axis name
        """
        specie_names = self.generateSpecieNames()
        specie_name = specie_names[index]
        return specie_name

    def generateQuantityNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected quantity names for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        quantity_names = self.getAxisQuantityNames(include_none=include_none)
        if quantity_names is None:
            axis_quantity_frame = self.getAxisQuantityFrame()
            window_runner = axis_quantity_frame.getWindowRunner()

            quantity_names = []
            axis_quantity_rows: List[AxisQuantityRow] = axis_quantity_frame.getAxisQuantityRows()
            for axis_quantity_row in axis_quantity_rows:
                quantity_element = axis_quantity_row.getAxisQuantityElement()
                quantity_name_key = getKeys(quantity_element)
                quantity_name = window_runner.getValue(quantity_name_key)
                quantity_names.append(quantity_name)

            self.quantity_names = quantity_names.copy()
            quantity_names = self.getAxisQuantityNames(include_none=include_none)

        return quantity_names

    def generateAxisQuantityName(
        self,
        index: int
    ) -> str:
        """
        Get selected quantity name for subaxis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        :param index: index of axis quantity per axis name
        """
        quantity_names = self.generateQuantityNames()
        quantity_name = quantity_names[index]
        return quantity_name

    def generateEnvelopeName(self) -> str:
        """
        Get name of envelope (e.g. "Amplitude") for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        """
        envelope_name = self.getEnvelopeName()
        if envelope_name is None:
            exists_nontimelike = self.existsNontimelikeSpecies()

            if exists_nontimelike:
                envelope_name = "None"
            else:
                axis_quantity_frame = self.getAxisQuantityFrame()
                envelope_radio_group = axis_quantity_frame.getAxisEnvelopeGroup()
                if envelope_radio_group is not None:
                    envelope_name = envelope_radio_group.getChosenRadioLabel()
                else:
                    envelope_name = "None"

            self.envelope_name = envelope_name

        return envelope_name

    def generateFunctionalName(self) -> str:
        """
        Get name of functional for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        """
        functional_name = self.getFunctionalName()
        if functional_name is None:
            exists_nontimelike = self.existsNontimelikeSpecies()

            if exists_nontimelike:
                functional_name = "None"
            else:
                axis_quantity_frame = self.getAxisQuantityFrame()
                functional_element = axis_quantity_frame.getAxisFunctionalElement()

                if functional_element is not None:
                    window_runner = axis_quantity_frame.getWindowRunner()
                    functional_key = getKeys(functional_element)
                    functional_name = window_runner.getValue(functional_key)
                else:
                    functional_name = "None"

            self.functional_name = functional_name

        return functional_name

    def generateTransformName(self) -> str:
        """
        Get name of transform for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        """
        transform_name = self.getTransformName()
        if transform_name is None:
            exists_nontimelike = self.existsNontimelikeSpecies()

            if exists_nontimelike:
                transform_name = "None"
            else:
                axis_quantity_frame = self.getAxisQuantityFrame()
                transform_element = axis_quantity_frame.getAxisTransformElement()

                if transform_element is not None:
                    window_runner = axis_quantity_frame.getWindowRunner()
                    transform_key = getKeys(transform_element)
                    transform_name = window_runner.getValue(transform_key)
                else:
                    transform_name = "None"

            self.transform_name = transform_name

        return transform_name

    def generateComplexName(self) -> str:
        """
        Get name of complex-reduction method for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        """
        complex_name = self.getComplexName()
        if complex_name is None:
            exists_nontimelike = self.existsNontimelikeSpecies()

            if exists_nontimelike:
                complex_name = "Real"
            else:
                axis_quantity_frame = self.getAxisQuantityFrame()
                complex_radio_group = axis_quantity_frame.getAxisComplexGroup()
                if complex_radio_group is not None:
                    complex_name = complex_radio_group.getChosenRadioLabel()
                else:
                    complex_name = "Real"

            self.complex_name = complex_name

        return complex_name

    def generateNormalizeNames(self) -> List[str]:
        """
        Get name of parameters to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        """
        normalize_names = self.getNormalizeNames()
        if normalize_names is None:
            normalize_names = []

            axis_quantity_frame = self.getAxisQuantityFrame()
            normalize_checkbox_group = axis_quantity_frame.getAxisNormalizeGroup()
            if normalize_checkbox_group is not None:
                checked_checkboxes = normalize_checkbox_group.getCheckedCheckboxes()
                for checked_checkbox in checked_checkboxes:
                    checkbox_attributes = vars(checked_checkbox)
                    is_disabled = checkbox_attributes["Disabled"]
                    if not is_disabled:
                        other_axis_name = checkbox_attributes["Text"]
                        normalize_names.append(other_axis_name)

            self.normalize_names = normalize_names

        return normalize_names

    def generateFunctionalFunctionalNames(
        self,
        include_none: bool = True
    ) -> List[str]:
        """
        Get names of functionals for axis.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve name from
        :param include_none: set True to include "None" functional. Set False otherwise.
        """
        functional_names = self.getFunctionalFunctionalNames(include_none=include_none)
        if functional_names is None:
            functional_names = []

            exists_nontimelike = self.existsNontimelikeSpecies()
            if not exists_nontimelike:
                axis_quantity_frame = self.getAxisQuantityFrame()
                parameter_functional_rows = axis_quantity_frame.getParameterFunctionalRows()
                if parameter_functional_rows is not None:
                    window_runner = axis_quantity_frame.getWindowRunner()
                    for parameter_functional_row in parameter_functional_rows:
                        functional_element = parameter_functional_row.getParameterFunctionalElement()
                        functional_key = getKeys(functional_element)
                        functional_name = window_runner.getValue(functional_key)
                        functional_names.append(functional_name)

            self.functional_names = functional_names.copy()
            functional_names = self.getFunctionalFunctionalNames(include_none=include_none)

        return functional_names

    def generateFunctionalParameterNamess(
        self,
        include_none: bool = True
    ) -> List[List[str]]:
        """
        Get name of axes to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        :param include_none: set True to include "None" functional. Set False otherwise.
        """
        parameter_namess = self.getFunctionalParameterNamess(include_none=include_none)
        if parameter_namess is None:
            parameter_namess = []

            exists_nontimelike = self.existsNontimelikeSpecies()
            if not exists_nontimelike:
                axis_quantity_frame = self.getAxisQuantityFrame()
                parameter_functional_rows = axis_quantity_frame.getParameterFunctionalRows()
                if parameter_functional_rows is not None:
                    for parameter_functional_row in parameter_functional_rows:
                        parameter_group = parameter_functional_row.getParameterFunctionalGroup()

                        parameter_names = []
                        checked_checkboxes = parameter_group.getCheckedCheckboxes()
                        for checked_checkbox in checked_checkboxes:
                            checkbox_attributes = vars(checked_checkbox)
                            checkbox_parameter_name = checkbox_attributes["Text"]
                            parameter_names.append(checkbox_parameter_name)

                        parameter_namess.append(parameter_names)

            self.parameter_namess = parameter_namess.copy()
            parameter_namess = self.getFunctionalParameterNamess(include_none=include_none)

        return parameter_namess

    def generateQuantityType(self) -> str:
        """
        Get type of quantity corresponding to axis-quantity.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantity` to retrieve type from
        :returns: "nonelike", "timelike", "parameterlike", or "functionallike"
        """
        quantity_type = self.getQuantityType()
        if quantity_type is None:
            exists_timelike_subaxis = self.existsLikeSpecies("timelike")
            exists_parameterlike_subaxis = self.existsLikeSpecies("parameterlike")
            functional_name = self.generateFunctionalName()

            specie_names = self.generateSpecieNames()
            specie_count = len(specie_names)
            nonelike_count = PlotQuantities.getLikeCount(specie_names, "nonelike")
            nonnonelike_count = specie_count - nonelike_count

            is_transformed = self.isTransformed()

            if exists_timelike_subaxis and exists_parameterlike_subaxis:
                quantity_type = "nonelike"
            elif not is_transformed and nonnonelike_count != 1:
                quantity_type = "nonelike"
            elif functional_name != "None":
                quantity_type = "functionallike"
            elif exists_timelike_subaxis:
                quantity_type = "timelike"
            elif exists_parameterlike_subaxis:
                parameterlike_count_subaxis = self.getLikeCount("parameterlike")
                if parameterlike_count_subaxis == 1:
                    quantity_type = "parameterlike"
                elif parameterlike_count_subaxis >= 2:
                    quantity_type = "nonelike"
            else:
                quantity_type = "nonelike"

            self.quantity_type = quantity_type

        return quantity_type


class PlotQuantities:
    specie_types = ["timelike", "parameterlike", "nonelike"]
    timelike_species = ["Variable", "Function"]
    parameterlike_species = ["Parameter"]
    nonelike_species = ["None"]

    def __init__(
        self,
        axis_quantity_frames: Union[AxisQuantityFrame, List[AxisQuantityFrame]]
    ) -> None:
        if isinstance(axis_quantity_frames, AxisQuantityFrame):
            axis_quantity_frames = [axis_quantity_frames]
        assert isinstance(axis_quantity_frames, list)

        axis_name2quantity = {}
        for axis_quantity_frame in axis_quantity_frames:
            assert isinstance(axis_quantity_frame, AxisQuantityFrame)

            axis_name = axis_quantity_frame.getName()
            axis_quantity = AxisQuantity(axis_quantity_frame)
            axis_name2quantity[axis_name] = axis_quantity
        self.axis_name2quantity = axis_name2quantity

        self.resetMetadata()

    def resetMetadata(self) -> None:
        """
        Reset attribute in object.

        :param self: :class:`~Layout.SimulationWindowRunner.PlotQuantities` to reset attributes for
        """
        axis_quantities: List[AxisQuantity] = self.getAxisQuantities()
        for axis_quantity in axis_quantities:
            axis_quantity.reset()

        self.valid_axis_names = None
        self.parameter_names = None

    def generateMetadata(self) -> None:
        """
        Generate all metadata attributes for axis-quantities contained in plot-quantities.

        :param self: class:`~Layout.AxisQuantity.PlotQuantities` to generate metadata in
        """
        axis_quantities: List[AxisQuantity] = self.getAxisQuantities()
        for axis_quantity in axis_quantities:
            axis_quantity.generateMetadata()

        self.generateValidAxisNames()
        self.generateParameterNames()

    def getAxisQuantities(
        self,
        names: Union[str, Iterable[str]] = None,
    ) -> Union[AxisQuantity, List[AxisQuantity]]:
        """
        Get axis quantity(s) corresponding to given axis name(s).

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve quantity(s) from
        :param names: name(s) of axis(es) to retrieve quantities for.
            Defaults to retrieving all axis quantities.
        """
        axis_quantities = self.axis_name2quantity

        def get(axis_name: str) -> AxisQuantity:
            """Base method for :meth:`~Layout.AxisQuantity.PlotQuantities.getAxisQuantities`"""
            axis_quantity = axis_quantities[axis_name]
            return axis_quantity

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=list,
            default_args=self.getAxisNames()
        )

    def getAxisQuantityFrames(
        self,
        names: Union[str, List[str]] = None,
    ) -> Union[AxisQuantityFrame, List[AxisQuantityFrame]]:
        """
        Get axis-quantity frames to determine desired property of plot quantities.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve frames from
        :param names: name(s) of axis-quantity frame(s).
            Defaults to names of all frames.
        """
        name2frame = self.name2frame

        def get(name: str):
            return name2frame[name]

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=dict,
            default_args=list(name2frame.keys())
        )

    def getAxisNames(self) -> List[str]:
        """
        Get names of axes to consider for plot quantities.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        """
        return list(self.axis_name2quantity.keys())

    def generateValidAxisNames(self) -> List[str]:
        """
        Get names of axes that require plotting a valid quantity.
        Ignore axes that are to be left empty.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        """
        valid_axis_names = self.valid_axis_names

        if valid_axis_names is None:
            axis_quantities: List[AxisQuantity] = self.getAxisQuantities()
            valid_axis_names = []
            for axis_quantity in axis_quantities:
                quantity_type = axis_quantity.generateQuantityType()
                if quantity_type != "nonelike":
                    axis_name = axis_quantity.getAxisName()
                    valid_axis_names.append(axis_name)

            self.valid_axis_names = valid_axis_names

        return valid_axis_names

    def generateParameterNames(self) -> List[str]:
        """
        Get names of parameters involved in plot.

        :param self: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve names from
        """
        parameter_names = self.parameter_names
        if parameter_names is None:
            parameter_names = []
            axis_quantities: List[AxisQuantity] = self.getAxisQuantities()
            for axis_quantity in axis_quantities:
                quantity_type = axis_quantity.generateQuantityType()
                if quantity_type == "parameterlike":
                    quantity_names = axis_quantity.generateQuantityNames(include_none=False)
                    assert len(quantity_names) == 1
                    parameter_name = quantity_names[0]
                    parameter_names.append(parameter_name)

            self.parameter_names = parameter_names

        assert isinstance(parameter_names, list)
        return parameter_names

    @classmethod
    def getSpecies(cls, like: str = None) -> List[str]:
        """
        Get collection of quantity species that may be treated as over-time.

        :param cls: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve species from
        :param like: species type to retrieve collection of species of.
            Can be "timelike", "parameterlike", "nonelike".
            Defaults to all species.
        """
        if like == "timelike":
            return cls.timelike_species
        elif like == "parameterlike":
            return cls.parameterlike_species
        elif like == "nonelike":
            return cls.nonelike_species
        elif like == None:
            all_species = []
            for specie_type in cls.specie_types:
                species_of_type = cls.getSpecies(specie_type)
                all_species.extend(species_of_type)
            return all_species
        else:
            raise ValueError("like must be 'timelike', 'parameterlike', or 'nonelike'")

    @classmethod
    def getLikeCount(
        cls,
        specie_names: List[str],
        like: str
    ):
        """
        Get number of given specie names within like-species group.

        :param cls: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve like-species from
        :param specie_names: names of species to count like-type in
        :param like: type of species to check for in collection of species.
            See :meth:`~Layout.AxisQuantity.PlotQuantities.getSpecies`.
        """
        like_species = cls.getSpecies(like)
        like_count = 0
        for specie_name in specie_names:
            if specie_name in like_species:
                like_count += 1

        return like_count

    @classmethod
    def existsLikeSpecies(
        cls,
        specie_names: List[str],
        like: str
    ):
        """
        Get whether at least one specie is of like-type.

        :param cls: :class:`~Layout.AxisQuantity.PlotQuantities` to retrieve like-species from
        :param specie_names: names of species to check for like-type in
        :param like: type of species to check for in collection of species.
            See :meth:`~Layout.AxisQuantity.PlotQuantities.getSpecies`.
        """
        like_species = cls.getSpecies(like)

        for specie_name in specie_names:
            if specie_name in like_species:
                return True

        return False

    def getPlotType(self) -> str:
        grid_axis_names = getAxisNames("grid")
        cartesian_axis_names = getAxisNames("cartesian")
        colorbar_axis_name = getAxisNames("color")[0]

        axis_names = self.getAxisNames()

        is_timelike = {}
        is_functionallike = {}
        is_parameterlike = {}
        is_nonelike = {}
        for axis_name in axis_names:
            axis_quantity = self.getAxisQuantities(names=axis_name)
            is_parameterlike[axis_name] = axis_quantity.isParameterLike()
            is_functionallike[axis_name] = axis_quantity.isFunctionalLike()
            is_timelike[axis_name] = axis_quantity.isTimeLike()
            is_nonelike[axis_name] = axis_quantity.isNoneLike()

        timelike_count = sum(is_timelike.values())
        exists_timelike = timelike_count >= 1
        multiple_timelikes = timelike_count >= 2
        exists_exactly_one_timelike = exists_timelike and not multiple_timelikes

        functionallike_count = sum(is_functionallike.values())
        exists_functionallike = functionallike_count >= 1

        timelike_and_functionallike = exists_timelike and exists_functionallike
        timelike_nor_functionallike = not exists_timelike and not exists_functionallike

        parameter_names = self.generateParameterNames()
        parameterlike_count = len(parameter_names)
        exists_parameterlike = parameterlike_count >= 1
        functionallike_and_not_parameterlike = exists_functionallike and not exists_parameterlike

        x_is_nonelike = is_nonelike['x']
        y_is_nonelike = is_nonelike['y']
        x_or_y_is_nonelike = x_is_nonelike or y_is_nonelike

        if x_or_y_is_nonelike:
            plot_type = ''
        elif exists_exactly_one_timelike:
            plot_type = ''
        elif timelike_and_functionallike:
            plot_type = ''
        elif timelike_nor_functionallike:
            plot_type = ''
        elif functionallike_and_not_parameterlike:
            plot_type = ''
        else:
            multiple_functionals = functionallike_count >= 2

            exists_cartesian_parameterlike = False
            cartesian_axis_quantities: List[AxisQuantity] = self.getAxisQuantities(names=cartesian_axis_names)
            for cartesian_axis_quantity in cartesian_axis_quantities:
                cartesian_axis_is_parameterlike = cartesian_axis_quantity.isParameterLike()
                if cartesian_axis_is_parameterlike:
                    exists_cartesian_parameterlike = True
                    break

            grid_parameterlike_count = 0
            grid_axis_quantities: List[AxisQuantity] = self.getAxisQuantities(names=grid_axis_names)
            for grid_axis_quantity in grid_axis_quantities:
                grid_axis_is_parameterlike = grid_axis_quantity.isParameterLike()
                if grid_axis_is_parameterlike:
                    grid_parameterlike_count += 1

            exists_grid_parameterlike = grid_parameterlike_count >= 1

            nongrid_parameterlike_count = parameterlike_count - grid_parameterlike_count
            multiple_nongrid_parameterlikes = nongrid_parameterlike_count >= 2

            if is_parameterlike['c']:
                if exists_timelike:
                    plot_type = 'nc'
                elif exists_functionallike:
                    if multiple_nongrid_parameterlikes:
                        plot_type = 'nc'
                    elif exists_parameterlike:
                        plot_type = 'nt'
            elif is_functionallike['c']:
                assert exists_cartesian_parameterlike
                if multiple_functionals:
                    if multiple_nongrid_parameterlikes:
                        plot_type = 't'
                    else:
                        plot_type = 'nt'
                elif exists_functionallike:
                    plot_type = 'c'
            elif is_timelike['c']:
                plot_type = 'nt'
            elif is_nonelike['c']:
                plot_type = ''

            for cartesian_axis_name in cartesian_axis_names:
                if is_parameterlike[cartesian_axis_name]:
                    if exists_functionallike:
                        if multiple_nongrid_parameterlikes:
                            plot_type += f"n{cartesian_axis_name:s}"
                        elif exists_parameterlike:
                            plot_type += cartesian_axis_name
                    elif exists_timelike:
                        plot_type += f"n{cartesian_axis_name:s}"
                elif is_functionallike[cartesian_axis_name]:
                    if is_parameterlike['c']:
                        plot_type += cartesian_axis_name
                    elif multiple_nongrid_parameterlikes:
                        plot_type += cartesian_axis_name
                    else:
                        assert exists_parameterlike
                        if multiple_functionals:
                            plot_type += cartesian_axis_name
                        else:
                            plot_type += f"n{cartesian_axis_name:s}"
                elif is_timelike[cartesian_axis_name]:
                    plot_type += cartesian_axis_name
                elif is_nonelike[cartesian_axis_name]:
                    pass

            if is_functionallike['C']:
                plot_type += 'C'

            if exists_grid_parameterlike:
                plot_type += '_'
                for grid_axis_name in grid_axis_names:
                    if is_parameterlike[grid_axis_name]:
                        plot_type += grid_axis_name
                    elif is_nonelike[grid_axis_name]:
                        pass

        return plot_type


class AxisQuantityFrameContainer:
    def __init__(self, axis_quantity_frames: List[AxisQuantityFrame] = None):
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityFrameContainer`.

        :param axis_quantity_frames: collection of axis-quantity frames included in window.
            Defaults to empty list.
        """
        if axis_quantity_frames is None:
            self.axis_quantity_frames = []
        else:
            assert isinstance(axis_quantity_frames, list)
            for axis_quantity_frame in axis_quantity_frames:
                assert isinstance(axis_quantity_frame, AxisQuantityFrame)
            self.axis_quantity_frames = axis_quantity_frames

    def addAxisQuantityFrame(self, axis_quantity_frame: AxisQuantityFrame) -> AxisQuantityFrame:
        """
        Add axis-quantity frame to object.
        Throws error if adding frame with duplicate name.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrameContainer` to add frame into
        :param axis_quantity_frame: frame to add into object
        """
        assert isinstance(self, AxisQuantityTabGroup)

        axis_quantity_name = axis_quantity_frame.getName()
        old_axis_quantity_frames: List[AxisQuantityFrame] = self.getAxisQuantityFrames()
        for old_axis_quantity_frame in old_axis_quantity_frames:
            old_axis_quantity_name = old_axis_quantity_frame.getName()
            if axis_quantity_name == old_axis_quantity_name:
                raise ValueError(f"AxisQuantityFrame {axis_quantity_name:s} already added")

        self.axis_quantity_frames.append(axis_quantity_frame)

    def getAxisQuantityFrames(
        self,
        names: Union[str, List[str]] = None,
    ) -> Union[AxisQuantityFrame, List[AxisQuantityFrame]]:
        """
        Get axis-quantity frame(s) in object.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrameContainer` to retrieve frame(s) from
        :param names: name(s) of frame(s) to retrieve
        """
        axis_quantity_frames: List[AxisQuantityFrame] = self.axis_quantity_frames
        axis_quantity_names = []
        name2frame = {}
        for axis_quantity_frame in axis_quantity_frames:
            axis_quantity_name = axis_quantity_frame.getName()
            axis_quantity_names.append(axis_quantity_name)
            name2frame[axis_quantity_name] = axis_quantity_frame

        def get(name: str):
            """Base method for :meth:`~Layout.AxisQuantity.AxisQuantityFrameContainer.getAxisQuantityFrames`"""
            return name2frame[name]

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=list,
            default_args=axis_quantity_names
        )


class AxisQuantityElement:
    def __init__(
        self,
        quantity_count_per_axis: int = 2,
        transform_names: List[str] = None,
        envelope_names: List[str] = None,
        functional_names: List[str] = None,
        complex_names: List[str] = None,
        normalize_parameter_names: List[str] = None,
        include_none: bool = True,
        include_continuous: bool = True,
        include_discrete: bool = True,
        include_scalefactor: bool = True,
        get_plot_choices: Callable[[], List[str]] = None,
        window: SimulationWindow = None
    ):
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityElement`.

        :param quantity_count_per_axis: number of possible quantities to select per axis
        :param transform_names: collection of names for math transforms to perform on results.
            Defaults to empty list.
        :param envelope_names: collection of names for envelopes to perform on results.
            Defaults to empty list.
        :param functional_names: collection of names for functionals to perform on results.
            Defaults to empty list.
        :param complex_names: collection of names for complex-reduction methods to perform on results.
            Defaults to empty list.
        :param normalize_parameter_names: names of parameters to normalize data over.
            Defaults to empty list;
        :param include_normalize: set True to include element to normalize data.
            Set False otherwise.
        :param include_none: set True to allow user to choose "None" for quantity.
            Set False otherwise.
        :param include_continuous: set True to allow user to choose continuous-like quantities.
            Set False otherwise.
        :param include_discrete: set True to allow user to choose discrete-like quantities.
            Set False otherwise.
        :param include_discrete: set True to allow user to choose scale factor to proportional quantities.
            Set False otherwise.
        :param get_plot_choices: callable acting similar to dictionary indicating quantity choices to plot along axis.
            Key (argument name "species" in callable) is specie of choice (e.g. "Variable", "Parameter").
            Value is list of quantity names corresponding to specie.
            Must be provided when window is not provided.
        """

        if get_plot_choices is not None:
            assert callable(get_plot_choices)
            self.getPlotChoices = get_plot_choices
        else:
            getPlotChoices = getattr(window, "getPlotChoices", None)
            assert callable(getPlotChoices)
            self.getPlotChoices = getPlotChoices

        self.quantity_count_per_axis = quantity_count_per_axis
        self.include_none = include_none
        self.include_continuous = include_continuous
        self.include_discrete = include_discrete
        self.include_scalefactor = include_scalefactor

        self.transform_names = [] if transform_names is None else transform_names
        self.envelope_names = [] if envelope_names is None else envelope_names
        self.functional_names = [] if functional_names is None else functional_names
        self.complex_names = [] if complex_names is None else complex_names

        self.normalize_parameter_names = [] if normalize_parameter_names is None else normalize_parameter_names

    def getQuantityCountPerAxis(self) -> int:
        """
        Get number of possible quantities to select per axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve count from
        """
        return self.quantity_count_per_axis

    def getTransformNames(self) -> List[str]:
        """
        Get names of transforms.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve names from
        """
        return self.transform_names

    def getEnvelopeNames(self) -> List[str]:
        """
        Get names of envelope.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve names from
        """
        return self.envelope_names

    def getFunctionalNames(self) -> List[str]:
        """
        Get names of functionals.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve names from
        """
        return self.functional_names

    def getComplexNames(self) -> List[str]:
        """
        Get names of complex-reduction methods.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve names from
        """
        return self.complex_names

    def getNormalizeParameterNames(self) -> List[str]:
        """
        Get names of parameters to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve names from
        """
        return self.normalize_parameter_names

    def includeScaleFactor(self) -> bool:
        """
        Get whether to include element to scale data along axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve boolean from
        """
        return self.include_scalefactor

    def includeNone(self) -> bool:
        """
        Get whether to include "None" choice for quantity.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve boolean from
        """
        return self.include_none

    def includeContinuous(self) -> bool:
        """
        Get whether to include continuous (timelike) choice(s) for quantity.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve boolean from
        """
        return self.include_continuous

    def includeDiscrete(self) -> bool:
        """
        Get whether to include dicrete (parameterlike) choice(s) for quantity.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityElement` to retrieve boolean from
        """
        return self.include_discrete


class AxisQuantityWindow(AxisQuantityFrameContainer):
    """
    :ivar axis_names: dictionary of axis names included in window.
        Key is type of axis (e.g. "Cartesian", "Color", "Grid").
        Value is tuple of corresponding axis names for type.
    """

    def __init__(
        self,
        axis_type2names: Dict[str, Tuple[str]],
        axis_quantity_frames: List[AxisQuantityFrame]
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityWindow`.

        :param axis_type2names: dictionary of axis names included in window.
            Key is type of axis (e.g. "cartesian", "color", "grid").
            Value is tuple of corresponding axis names for type.
        :param axis_quantity_frames: see :class:`~Layout.AxisQuantity.AxisQuantityFrameContainer`
        """
        assert isinstance(axis_quantity_frames, list)
        for axis_quantity_frame in axis_quantity_frames:
            assert isinstance(axis_quantity_frame, AxisQuantityFrame)

        AxisQuantityFrameContainer.__init__(
            self,
            axis_quantity_frames=axis_quantity_frames
        )

        assert isinstance(axis_type2names, dict)
        for axis_type, axis_type_names in axis_type2names.items():
            assert isinstance(axis_type, str)
            assert isinstance(axis_type_names, tuple)
            for axis_specie_name in axis_type_names:
                assert isinstance(axis_specie_name, str)
        self.axis_type2names = axis_type2names

        self.plot_quantities = PlotQuantities(axis_quantity_frames)

    def getAxisNames(self, filter: str = None) -> List[str]:
        """
        Get names for axes.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityWindow` to retrieve names from
        :param filter: only include axis types that satisfy this filter (e.g. "cartesian", "grid", or "color")
        :returns: List of axis names (e.g. ['x', 'y', 'c'])
        """
        return getAxisNames(
            filter=filter,
            axis_type2names=self.axis_type2names
        )


class AxisQuantityWindowRunner:
    def __init__(
        self,
        axis_quantity_frames: Union[AxisQuantityFrame, List[AxisQuantityFrame]]
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityWindowRunner`.
        """
        self.plot_quantities = PlotQuantities(axis_quantity_frames)

    def getPlotQuantities(self) -> PlotQuantities:
        """
        Get object to determine plot quantities in window.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityWindowRunner` to retrieve object from
        """
        return self.plot_quantities

    def updatePlotChoices(
        self,
        names: Union[str, List[str]] = None,
    ) -> None:
        """
        Update plot choices for desired axis(es).
        This allows user to select new set of quantities to plot.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityWindowRunner` to change plot choices in
        :param names: name(s) of axis(es) to update choices for.
            Updates all axes by default.
        """
        plot_quantities = self.getPlotQuantities()

        def update(name: str) -> None:
            """Base method for :meth:`~Layout.AxisQuantity.SimulationWindowRunner.updatePlotChoices`"""
            axis_quantity = plot_quantities.getAxisQuantities(names=name)
            axis_quantity.updatePlotChoices()

        axis_names = plot_quantities.getAxisNames()
        return recursiveMethod(
            args=names,
            base_method=update,
            valid_input_types=str,
            default_args=axis_names
        )


class AxisQuantityTabGroup(
    TabGroup,
    AxisQuantityElement,
    AxisQuantityFrameContainer
):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ):
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityTabGroup`.

        :param name: name of tab
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that tab is stored in
        :param kwargs: additional arguments to input into :class:`~Layout.AxisQuantity.AxisQuantityElement`
        """
        AxisQuantityElement.__init__(
            self,
            window=window,
            **kwargs
        )
        AxisQuantityFrameContainer.__init__(self)

        quantity_count_per_axis = self.getQuantityCountPerAxis()
        transform_names = self.getTransformNames()
        envelope_names = self.getEnvelopeNames()
        functional_names = self.getFunctionalNames()
        complex_names = self.getComplexNames()

        parameter_names = self.getPlotChoices(species="Parameter")

        axis_names = ('x', 'y', 'z', 'c', 'C', 'X', 'Y')
        numerical_axis_names = ('x', 'y', 'z', 'c', 'C')
        normalize_parameter_names = parameter_names
        standard_tab_frames = []
        nonstandard_tab_frames = []
        for axis_name in axis_names:
            if axis_name in numerical_axis_names:
                axis_quantity_count = quantity_count_per_axis
                axis_transform_names = transform_names
                axis_envelope_names = envelope_names
                axis_functional_names = functional_names
                axis_complex_names = complex_names
                axis_include_continuous = True
                axis_normalize_parameter_names = normalize_parameter_names
                axis_include_scalefactor = True
            else:
                axis_quantity_count = 1
                axis_transform_names = []
                axis_envelope_names = []
                axis_functional_names = []
                axis_complex_names = []
                axis_include_continuous = False
                axis_normalize_parameter_names = []
                axis_include_scalefactor = False

            axis_include_none = axis_name not in ('x', 'y')
            axis_include_discrete = axis_name not in ('C', )

            getPlotChoices = window.getPlotChoices
            axis_quantity_frame = AxisQuantityFrame(
                axis_name,
                window=window,
                get_plot_choices=getPlotChoices,
                quantity_count_per_axis=axis_quantity_count,
                include_none=axis_include_none,
                include_continuous=axis_include_continuous,
                include_discrete=axis_include_discrete,
                normalize_parameter_names=axis_normalize_parameter_names,
                include_scalefactor=axis_include_scalefactor,
                transform_names=axis_transform_names,
                envelope_names=axis_envelope_names,
                functional_names=axis_functional_names,
                complex_names=axis_complex_names
            )
            self.addAxisQuantityFrame(axis_quantity_frame)

            if axis_name in ('x', 'y', 'z', 'c'):
                standard_tab_frames.append(axis_quantity_frame)
            else:
                nonstandard_tab_frames.append(axis_quantity_frame)

        standard_tab = AxisQuantityTab(
            "Standard",
            window,
            axis_quantity_frames=standard_tab_frames
        )
        nonstandard_tab = AxisQuantityTab(
            "Non-Standard",
            window,
            axis_quantity_frames=nonstandard_tab_frames
        )

        tabs = [
            standard_tab,
            nonstandard_tab
        ]
        TabGroup.__init__(self, tabs, name=name)


class AxisQuantityTab(Tab):
    """
    This class contains the layout for the plotting tab in the simulation window.
        # . Header row to identify functions for input
        # . Axis name label for each axis to identify which axis input affects
        # . Combobox to input quantity species for each axis
        # . Combobox to input quantity for each axis
        # . Combobox to input transform type for each axis
    """

    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        axis_quantity_frames: List[AxisQuantityFrame]
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityTab`.

        :param name: name of tab
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that tab is stored in
        :param frames: frames contained in tab
        """
        Tab.__init__(self, name, window)

        self.axis_quantity_frames = axis_quantity_frames

    def getFrames(self) -> List[AxisQuantityFrame]:
        """
        Get axis-quantity frames stored in tab.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityTab` to retrieve frames from
        """
        return self.axis_quantity_frames

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for standard-axis plotting tab.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityTab` to retrieve layout from
        """
        frame_objs = self.getFrames()
        frames = [
            frame_obj.getFrame()
            for frame_obj in frame_objs
        ]

        frame_layout = [
            [frame]
            for frame in frames
        ]
        layout = [[sg.Column(
            frame_layout,
            scrollable=True,
            vertical_scroll_only=True,
            size=(None, 600)
        )]]

        return layout


class AxisQuantityFrame(Frame, AxisQuantityElement):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityFrame`.

        :param name: name of axis
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that frame is stored in
        :param kwargs: additional arguments to pass into :class:`~Layout.AxisQuantity.AxisQuantityElement`
        """
        Frame.__init__(self, name, window)
        AxisQuantityElement.__init__(
            self,
            window=window,
            **kwargs
        )

        parameter_names = self.getPlotChoices(species="Parameter")
        parameter_count = len(parameter_names)
        self.parameter_functional_count = min(2, parameter_count)
        assert self.parameter_functional_count <= parameter_count

        axis_name = self.getName()
        include_none = self.includeNone()
        include_continuous = self.includeContinuous()
        include_discrete = self.includeDiscrete()
        window_obj = self.getWindowObject()

        quantity_count_per_axis = self.getQuantityCountPerAxis()
        getPlotChoices = self.getPlotChoices
        self.axis_quantity_rows = []
        for index in range(quantity_count_per_axis):
            include_none_per_quantity = index != 0 or include_none
            axis_quantity_row = AxisQuantityRow(
                axis_name,
                window=window_obj,
                index=index,
                get_plot_choices=getPlotChoices,
                include_none=include_none_per_quantity,
                include_continuous=include_continuous,
                include_discrete=include_discrete
            )
            self.axis_quantity_rows.append(axis_quantity_row)

    def reset(self) -> None:
        """
        Reset properties for axis quantity to default.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to reset in
        """
        transform_element = self.getAxisTransformElement()
        if isinstance(transform_element, sg.Combo):
            transform_names = self.getTransformNames()
            default_transform_name = transform_names[0]
            transform_element.update(default_transform_name)

        functional_element = self.getAxisFunctionalElement()
        if isinstance(functional_element, sg.Combo):
            functional_names = self.getFunctionalNames()
            default_functional_name = functional_names[0]
            functional_element.update(default_functional_name)

        normalize_group = self.getAxisNormalizeGroup()
        if isinstance(normalize_group, CheckboxGroup):
            normalize_group.reset()

        complex_group = self.getAxisComplexGroup()
        if isinstance(complex_group, RadioGroup):
            complex_group.reset()

        envelope_group = self.getAxisEnvelopeGroup()
        if isinstance(envelope_group, RadioGroup):
            envelope_group.reset()

        scale_factor_element = self.getScaleFactorElement()
        if isinstance(scale_factor_element, sg.Spin):
            default_scale_factor = "1e0"
            scale_factor_element.update(default_scale_factor)

        axis_quantity_rows: List[AxisQuantityRow] = self.getAxisQuantityRows()
        for axis_quantity_row in axis_quantity_rows:
            axis_quantity_row.reset()

        parameter_functional_rows = self.getParameterFunctionalRows()
        for parameter_functional_row in parameter_functional_rows:
            parameter_functional_row.reset()

    def getParameterFunctionalCount(self) -> int:
        """
        Get max number of functionals to perform over parameters.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve number from
        """
        return self.parameter_functional_count

    @storeElement
    def getAxisTransformElement(self) -> sg.InputCombo:
        """
        Get element to take user input for tranform.
        This allows user to choose which transform to perform on plot quantities.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        transform_names = self.getTransformNames()
        if len(transform_names) >= 1:
            name = self.getName()

            return sg.InputCombo(
                values=transform_names,
                default_value=transform_names[0],
                tooltip=f"Choose transform to perform on {name:s}-axis of plot",
                enable_events=False,
                size=self.getDimensions(name="transform_type_combobox"),
                key=f"{cc_pre:s} TRANSFORM {name:s}_AXIS"
            )
        else:
            return None

    @storeElement
    def getAxisFunctionalElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis functional.
        This allows user to choose which type of functional to calculate for a plot quantity (e.g. frequency, amplitude).

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        functional_names = self.getFunctionalNames()
        if len(functional_names) >= 1:
            name = self.getName()

            default_value = functional_names[0]
            return sg.InputCombo(
                values=functional_names,
                default_value=default_value,
                tooltip=f"Choose functional to perform on {name:s}-axis of plot",
                enable_events=False,
                size=self.getDimensions(name="axis_functional_combobox"),
                key=f"-{cc_pre:s} FUNCTIONAL {name:s}_AXIS-"
            )
        else:
            return None

    @storeElement
    def getAxisNormalizeGroup(self) -> NormalizeCheckboxGroup:
        """
        Get element to take user input for whether to normalize axis data.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        normalize_parameter_names = self.getNormalizeParameterNames()
        if len(normalize_parameter_names) >= 1:
            name = self.getName()
            window_obj = self.getWindowObject()

            normalize_checkbox_group = NormalizeCheckboxGroup(
                name=name,
                other_axis_names=normalize_parameter_names,
                window=window_obj
            )
            return normalize_checkbox_group
        else:
            return None

    @storeElement
    def getAxisComplexGroup(self) -> ComplexRadioGroup:
        """
        Get element to take user input for complex-reduction method (e.g. "Magnitude", "Phase").

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        complex_names = self.getComplexNames()
        if len(complex_names) >= 1:
            axis_name = self.getName()
            window_obj = self.getWindowObject()

            complex_radio_group = ComplexRadioGroup(
                axis_name=axis_name,
                window=window_obj,
                complex_names=complex_names
            )
            return complex_radio_group
        else:
            return None

    @storeElement
    def getAxisEnvelopeGroup(self) -> EnvelopeRadioGroup:
        """
        Get element to take user input for envelope (e.g. "Amplitude").

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        envelope_names = self.getEnvelopeNames()
        if len(envelope_names) >= 1:
            axis_name = self.getName()
            window_obj = self.getWindowObject()

            envelope_radio_group = EnvelopeRadioGroup(
                axis_name=axis_name,
                window=window_obj,
                envelope_names=envelope_names
            )
            return envelope_radio_group
        else:
            return None

    @storeElement
    def getScaleFactorElement(self) -> sg.Spin:
        """
        Get element that allows user to input scale factor.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        include_scalefactor = self.includeScaleFactor()
        if include_scalefactor:
            name = self.getName()
            values = [f"1e{int(exponent):d}" for exponent in np.linspace(-24, 24, 49)]

            return sg.Spin(
                values=values,
                initial_value="1e0",
                tooltip=f"Choose scale factor for {name:s}-axis. Data is divided by this factor.",
                size=self.getDimensions(name="scale_factor_spin"),
                key=f"-SCALE FACTOR {name:s}_AXIS-"
            )
        else:
            return None

    def getAxisQuantityRows(
        self,
        indicies: int = None
    ) -> Union[AxisQuantityRow, List[AxisQuantityRow]]:
        """
        Get axis-quantity row(s) within frame.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve row from
        :param index: index(es) of row(s) within frame.
            Defaults to all rows within frame.
        """
        axis_quantity_rows = self.axis_quantity_rows

        def get(index: int):
            """Base method for :meth:`~Layout.AxisQuantity.AxisQuantityFrame.getAxisQuantityRows`"""
            axis_quantity_row = axis_quantity_rows[index]
            return axis_quantity_row

        return recursiveMethod(
            args=indicies,
            base_method=get,
            valid_input_types=int,
            output_type=list,
            default_args=range(len(axis_quantity_rows))
        )

    @storeElement
    def getParameterFunctionalRow(self, index: int) -> ParameterFunctionalRow:
        """
        Get element to take user input for functional over parameters.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        :param index: index of functional row within collection of functionals over parameters
        """
        parameter_names = self.getPlotChoices(species="Parameter")
        parameter_count = len(parameter_names)
        functional_names = self.getFunctionalNames()
        functional_count = len(functional_names)
        if parameter_count >= 1 and functional_count >= 1:
            name = self.getName()
            window_obj = self.getWindowObject()

            getPlotChoices = self.getPlotChoices
            functional_row = ParameterFunctionalRow(
                name,
                index,
                window=window_obj,
                get_plot_choices=getPlotChoices,
                functional_names=functional_names
            )
            return functional_row
        else:
            return None

    @storeElement
    def getParameterFunctionalRows(self) -> List[ParameterFunctionalRow]:
        """
        Get element to take user input for functionals over parameters.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        """
        count = self.parameter_functional_count

        parameter_names = self.getPlotChoices(species="Parameter")
        parameter_count = len(parameter_names)
        assert count <= parameter_count

        functional_names = self.getFunctionalNames()
        functional_count = len(functional_names)
        if functional_count >= 1:
            parameter_functional_rows = [
                self.getParameterFunctionalRow(index)
                for index in range(count)
            ]
        else:
            parameter_functional_rows = []

        return parameter_functional_rows

    @storeElement
    def getResetButton(self) -> sg.Button:
        axis_name = self.getName()
        return sg.Button(
            button_text="Reset",
            key=f"-{qr_pre:s} {axis_name:s}_AXIS-",
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get frame that allows user input for a single axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve frame from
        """
        rows = []

        envelope_group = self.getAxisEnvelopeGroup()
        if isinstance(envelope_group, RadioGroup):
            envelope_row_elements = [
                sg.Text("Envelope:"),
                *envelope_group.getRadios()
            ]
            rows.append(Row(elements=envelope_row_elements))

        transform_element = self.getAxisTransformElement()
        if isinstance(transform_element, sg.InputCombo):
            transform_row_elements = [
                sg.Text("Transform:"),
                transform_element
            ]
        else:
            transform_row_elements = []

        functional_element = self.getAxisFunctionalElement()
        if isinstance(functional_element, sg.InputCombo):
            functional_row_elements = [
                sg.Text("Functional:"),
                functional_element
            ]
        else:
            functional_row_elements = []

        row_elements = [
            *transform_row_elements,
            *functional_row_elements
        ]
        if len(row_elements) >= 1:
            rows.append(Row(elements=row_elements))

        complex_group = self.getAxisComplexGroup()
        if isinstance(complex_group, RadioGroup):
            complex_row_elements = [
                sg.Text("Complex:"),
                *complex_group.getRadios()
            ]
            rows.append(Row(elements=complex_row_elements))

        axis_quantity_rows = self.getAxisQuantityRows()
        rows.extend(axis_quantity_rows)

        scale_factor_element = self.getScaleFactorElement()
        if isinstance(scale_factor_element, sg.Spin):
            scale_factor_row_elements = [
                sg.Text("Scale Factor:"),
                scale_factor_element
            ]
        else:
            scale_factor_row_elements = []

        parameter_functional_rows = self.getParameterFunctionalRows()
        if len(parameter_functional_rows) >= 1:
            rows.extend(parameter_functional_rows)

        normalize_checkbox_group = self.getAxisNormalizeGroup()
        if isinstance(normalize_checkbox_group, CheckboxGroup):
            normalize_checkboxes = normalize_checkbox_group.getCheckboxes()
            normalize_row_elements = [
                sg.Text("Normalize:"),
                *normalize_checkboxes
            ]
        else:
            normalize_row_elements = []

        row_elements = [
            *normalize_row_elements,
            *scale_factor_row_elements
        ]
        if len(row_elements) >= 1:
            rows.append(Row(elements=row_elements))

        reset_button = self.getResetButton()
        reset_row = Row(elements=reset_button)
        rows.append(reset_row)

        layout = Layout(rows).getLayout()
        return layout


class ParameterFunctionalRow(Row, AxisQuantityElement):
    def __init__(
        self,
        axis_name: str,
        index: int,
        window: SimulationWindow,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.ParameterFunctionalRow`.

        :param axis_name: name of axis associated with row
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that group is stored in
        :param index: index of functional row within collection of functionals over parameters
        :param kwargs: additional arguments to pass into :class:`~Layout.AxisQuantity.AxisQuantityElement`
        :
        """
        name = axis_name + str(index)
        Row.__init__(self, name, window=window)
        AxisQuantityElement.__init__(
            self,
            window=window,
            **kwargs
        )

        self.axis_name = axis_name

        functional_element = self.getParameterFunctionalElement()
        parameter_group = self.getParameterFunctionalGroup()
        elements = [
            sg.Text("Functional:"),
            functional_element,
            sg.Text("Over:"),
            *parameter_group.getElements()
        ]

        if None not in elements:
            self.addElements(elements)

    def reset(self) -> None:
        """
        Reset elements in parameter-functional row.

        :param self: :class:`~Layout.AxisQuantity.ParameterFunctionalRow` to reset
        """
        window_obj = self.getWindowObject()
        window = window_obj.getWindow()

        functional_names = self.getFunctionalNames()
        default_functional = functional_names[0]
        functional_element = self.getParameterFunctionalElement()
        functional_key = getKeys(functional_element)
        functional_element.update(default_functional)
        window.write_event_value(functional_key, default_functional)

        parameter_group = self.getParameterFunctionalGroup()
        parameter_group.reset()

    def getAxisName(self) -> str:
        """
        Get name of axis.

        :param self: :class:`~Layout.AxisQuantity.ParameterFunctionRow` to retrieve name from
        """
        return self.axis_name

    @storeElement
    def getParameterFunctionalElement(self) -> sg.InputCombo:
        """
        Get element to take user input for a second functional over multiple parameters.
        This allows user to choose which type of functional to calculate for a plot quantity (e.g. frequency, amplitude).

        :param self: :class:`~Layout.AxisQuantity.ParameterFunctionalRow` to retrieve element from
        """
        functional_names = self.getFunctionalNames()
        if len(functional_names) >= 1:
            axis_name = self.getAxisName()
            name = self.getName()

            default_value = functional_names[0]
            return sg.InputCombo(
                values=functional_names,
                default_value=default_value,
                tooltip=f"Choose functional to perform on {axis_name:s}-axis of plot over chosen parameters",
                enable_events=False,
                size=self.getDimensions(name="axis_functional_combobox"),
                key=f"-{cc_pre:s} MULTIFUNCTIONAL {name:s}_AXIS-"
            )
        else:
            return None

    @storeElement
    def getParameterFunctionalGroup(self) -> ParameterFunctionalCheckboxGroup:
        """
        Get element to take user input for whether to normalize axis data.

        :param self: :class:`~Layout.AxisQuantity.ParameterFunctionalRow` to retrieve element from
        """
        parameter_names = self.getPlotChoices(species="Parameter")
        if len(parameter_names) >= 1:
            name = self.getName()
            window_obj = self.getWindowObject()

            functional_checkbox_group = ParameterFunctionalCheckboxGroup(
                name=name,
                parameter_names=parameter_names,
                window=window_obj
            )
            return functional_checkbox_group
        else:
            return None


class EnvelopeRadioGroup(RadioGroup):
    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        envelope_names: List[str] = None
    ):
        """
        Constructor for :class:`~Layout.AxisQuantity.EnvelopeRadioGroup`.

        :param axis_name: name of axis associated with group
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that group is stored in
        :param envelope_names: collection of names for envelopes (e.g. ["None", "Ampltude"]).
            Defaults to ["None"].
        """
        group_id = f"-ENVELOPE {axis_name:s}_AXIS-"

        RadioGroup.__init__(
            self,
            radios=[],
            name=axis_name,
            group_id=group_id,
            window=window
        )

        if envelope_names is None:
            self.envelope_names = ["None"]
        else:
            self.envelope_names = envelope_names

        for envelope_name in envelope_names:
            radio = self.getEnvelopeRadio(envelope_name)
            self.addElements(radio)

    @storeElement
    def getEnvelopeRadio(
        self,
        envelope_name: str,
        default_name: str = "None"
    ) -> sg.Radio:
        """
        Get element to take user input for envelope (individual).

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        :param envelope_name: name envelope type
        :param default_name: name of default radio to be set true
        """
        axis_name = self.getName()
        is_default = envelope_name == default_name
        radio_group_id = self.getGroupId()
        radio_key = radio_group_id + f"{envelope_name.upper():s}-"

        radio = sg.Radio(
            text=envelope_name,
            tooltip=f"Choose envelope to perform on {axis_name:s}-axis of plot",
            group_id=radio_group_id,
            default=is_default,
            key=radio_key
        )
        return radio


class ComplexRadioGroup(RadioGroup):
    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        complex_names: List[str] = None
    ):
        """
        Constructor for :class:`~Layout.AxisQuantity.ComplexRadioGroup`.

        :param axis_name: name of axis associated with group
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that group is stored in
        :param complex_names: collection of names for complex-reduction methods (e.g. "Magnitude", "Phase").
            Defaults to empty list.
        """
        group_id = f"-COMPLEX {axis_name:s}_AXIS-"

        RadioGroup.__init__(
            self,
            radios=[],
            name=axis_name,
            group_id=group_id,
            window=window
        )

        if complex_names is None:
            self.complex_names = []
        else:
            assert isinstance(complex_names, list)
            for complex_name in complex_names:
                assert isinstance(complex_name, str)
            self.complex_names = complex_names

        for complex_name in complex_names:
            radio = self.getComplexRadio(complex_name)
            self.addElements(radio)

    @storeElement
    def getComplexRadio(
        self,
        complex_name: str,
        default_name: str = "Real"
    ) -> sg.Radio:
        """
        Get element to take user input for complex system (individual).

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        :param complex_name: name of complex-reduction method
        :param default_name: name of default radio to be set true
        """
        axis_name = self.getName()
        is_default = complex_name == default_name
        radio_group_id = self.getGroupId()
        radio_key = f"{radio_group_id:s}{complex_name.upper():s}-"

        radio = sg.Radio(
            text=complex_name,
            tooltip=f"Choose complex-reduction method to perform on {axis_name:s}-axis of plot",
            group_id=radio_group_id,
            default=is_default,
            key=radio_key
        )
        return radio


class NormalizeCheckboxGroup(CheckboxGroup):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        other_axis_names: Iterable[str] = None
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup`.

        :param name: name of axis to normalize from
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that group is stored in
        :param other_axis_names: names of other axes to normalize over.
            Defaults to empty list.
        """
        if other_axis_names is None:
            self.other_axis_names = []
        else:
            assert isinstance(other_axis_names, Iterable)
            for other_axis_name in other_axis_names:
                assert isinstance(other_axis_name, str)
            self.other_axis_names = list(other_axis_names)

        CheckboxGroup.__init__(
            self,
            window=window,
            name=name
        )

        normalize_group_checkboxes = [
            self.getNormalizeElement(other_axis_name)
            for other_axis_name in other_axis_names
        ]
        self.addCheckboxes(normalize_group_checkboxes)

    def getOtherAxisNames(self) -> List[str]:
        """
        Get collection of other axis names to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup` to retrieve names from
        """
        return self.other_axis_names

    @storeElement
    def getNormalizeElement(self, name: str) -> sg.Checkbox:
        """
        Get element to allow user to normalize data over another axis.

        :param self: :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup` to retrieve element from
        :param name: name of axis to normalize data over (if chosen)
        """
        self_axis_name = self.getName()
        other_axis_name = name

        return sg.Checkbox(
            text=other_axis_name,
            default=False,
            disabled=False,
            key=f"-NORMALIZE {self_axis_name:s}_AXIS OVER {other_axis_name:s}_AXIS-"
        )


class ParameterFunctionalCheckboxGroup(CheckboxGroup):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        parameter_names: Iterable[str] = None
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup`.

        :param name: name of axis to normalize from
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that group is stored in
        :param parameter_names: names of parameters to perform functional over.
            Defaults to empty list.
        """
        if parameter_names is None:
            self.parameter_names = []
        else:
            assert isinstance(parameter_names, Iterable)
            for parameter_name in parameter_names:
                assert isinstance(parameter_name, str)
            self.parameter_names = list(parameter_names)

        CheckboxGroup.__init__(
            self,
            window=window,
            name=name
        )

        functional_group_checkboxes = [
            self.getParameterElement(parameter_name)
            for parameter_name in parameter_names
        ]
        self.addCheckboxes(functional_group_checkboxes)

    def getParameterNames(self) -> List[str]:
        """
        Get collection of other axis names to normalize data over.

        :param self: :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup` to retrieve names from
        """
        return self.parameter_names

    @storeElement
    def getParameterElement(self, name: str) -> sg.Checkbox:
        """
        Get element to allow user to normalize data over another axis.

        :param self: :class:`~Layout.AxisQuantity.AxisNormalizeCheckboxGroup` to retrieve element from
        :param name: name of parameter to perform functional over (if chosen)
        """
        axis_name = self.getName()
        parameter_name = name

        return sg.Checkbox(
            text=parameter_name,
            default=False,
            disabled=False,
            key=f"-NORMALIZE {axis_name:s}_AXIS OVER {parameter_name:s}_PARAMETER-"
        )


class AxisQuantityRow(AxisQuantityElement, Row):
    """
    :ivar getPlotChoices: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotChoices`
    """

    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        index: int = 0,
        include_none: bool = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.AxisQuantity.AxisQuantityRow`.

        :param axis_name: name of axis
        :param index: quantity index per axis
        :param window: :class:`~Layout.AxisQuantity.SimulationWindow` that row is stored in
        :param include_none: set True to allow user to choose "None" for quantity.
            Set False otherwise.
            Must be True if index != 0.
            Defaults to index != 0.
        :param kwargs: additional arguments to pass into :class:`~Layout.AxisQuantity.AxisQuantityElement`
        """
        if index != 0:
            assert include_none
        if include_none is None:
            include_none = index != 0
        else:
            include_none = index != 0 or include_none

        name = axis_name + str(index)
        AxisQuantityElement.__init__(
            self,
            window=window,
            include_none=include_none,
            **kwargs
        )
        Row.__init__(
            self,
            name,
            window=window
        )

        self.axis_name = axis_name
        self.index = index

        include_none = self.includeNone()
        include_continuous = self.includeContinuous()
        include_discrete = self.includeDiscrete()

        axis_quantity_species = []
        if include_none:
            axis_quantity_species.append("None")
        if include_continuous:
            axis_quantity_species.extend(["Variable", "Function"])
        if include_discrete:
            parameter_choices = self.getPlotChoices(species="Parameter")
            if len(parameter_choices) >= 1:
                axis_quantity_species.append("Parameter")
        self.axis_quantity_species = axis_quantity_species

        elements = [
            sg.Text("Species:"),
            self.getAxisQuantitySpeciesElement(),
            sg.Text("Quantity:"),
            self.getAxisQuantityElement()
        ]
        self.addElements(elements)

    def reset(self) -> None:
        """
        Reset elements in axis-quantity row.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to reset
        """
        window_obj = self.getWindowObject()
        window = window_obj.getWindow()
        include_none = self.includeNone()

        if include_none:
            default_specie = "None"
        else:
            axis_quantity_species = self.getAxisQuantitySpecies()
            default_specie = axis_quantity_species[0]

        specie_element = self.getAxisQuantitySpeciesElement()
        specie_key = getKeys(specie_element)
        specie_element.update(default_specie)
        window.write_event_value(specie_key, default_specie)

        if default_specie == "None":
            default_quantity = ''
        else:
            quantity_choices = self.getPlotChoices(species=default_specie)
            default_quantity = quantity_choices[0]
        quantity_element = self.getAxisQuantityElement()
        quantity_key = getKeys(quantity_element)
        quantity_element.update(default_quantity)
        window.write_event_value(quantity_key, default_quantity)

    def getAxisName(self) -> str:
        """
        Get name of axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to retrieve name from
        """
        return self.axis_name

    def getIndex(self) -> int:
        """
        Get index of row within axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to retrieve index from
        """
        return self.index

    @storeElement
    def getAxisQuantityElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity.
        This allows user to choose which quantity to plot on the axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to retrieve element from
        """
        name = self.getName()

        include_none = self.includeNone()
        if include_none:
            values = ['']
            disabled = True
        else:
            values = self.getPlotChoices(species="Variable")
            disabled = False

        sg_kwargs = {
            "tooltip": f"Choose quantity to display on {name:s}-axis of plot",
            "enable_events": True,
            "size": self.getDimensions(name="axis_quantity_combobox"),
            "key": f"-{cc_pre:s} QUANTITY {name:s}_AXIS-"
        }

        sg_kwargs["values"] = values
        sg_kwargs["default_value"] = sg_kwargs["values"][0]
        sg_kwargs["disabled"] = disabled

        elem = sg.InputCombo(**sg_kwargs)
        return elem

    def getAxisQuantitySpecies(self) -> List[str]:
        """
        Get names of species for axis quantities.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to retrieve names from
        """
        return self.axis_quantity_species

    @storeElement
    def getAxisQuantitySpeciesElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity type.
        This allows user to choose which type of quantity to plot on the axis.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityRow` to retrieve element from
        """
        name = self.getName()
        axis_quantity_species = self.getAxisQuantitySpecies()
        default_specie = axis_quantity_species[0]

        return sg.InputCombo(
            values=axis_quantity_species,
            default_value=default_specie,
            tooltip=f"Choose quantity species to display on {name:s}-axis of plot",
            enable_events=True,
            size=self.getDimensions(name="axis_quantity_species_combobox"),
            key=f"-{ccs_pre:s} {name:s}_AXIS-"
        )
