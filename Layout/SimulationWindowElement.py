from __future__ import annotations

from typing import Dict, List, Tuple, Union

import PySimpleGUI as sg

from Layout.Layout import (Element, Layout, RadioGroup, Row, Slider, Window,
                           storeElement)


class GroupIndexSlider(Slider):
    def __init__(
        self,
        window: Window,
        sample_size: int
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider`.

        :param window: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` that slider is stored in
        :param sample_size: number of simulation indicies to include in slider
        """
        assert isinstance(window, Window)
        Slider.__init__(
            self,
            window=window,
            minimum=1,
            maximum=sample_size,
            stepcount=sample_size
        )

        assert isinstance(sample_size, int)
        self.simulation_count = sample_size

        slider = self.getSlider()
        name_label = sg.Text("Sample Index:")
        minimum_label = sg.Text("1")
        maximum_label = sg.Text(f"{sample_size:d}")
        self.maximum_label = maximum_label

        elements = [
            name_label,
            minimum_label,
            slider,
            maximum_label
        ]
        self.addElements(elements)

    def getSimulationCount(self) -> int:
        """
        Get simulation count to include on slider.

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to retrieve count from
        """
        return self.simulation_count

    def updateSimulationCount(self, simulation_count: int) -> None:
        """
        Change simulation count to include on slider (e.g. if user selected new sample size).

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to change count in
        :param self: new number of simulation indicies to include in slider
        """
        assert isinstance(simulation_count, int)

        maximum_label = self.getMaximumLabel()
        maximum_label.update(simulation_count)

        slider = self.getSlider()
        minimum = self.getMinimum()
        slider.update(range=(minimum, simulation_count))

        self.simulation_count = simulation_count

    @storeElement
    def getSlider(self) -> sg.Slider:
        """
        Get slider to take user input for simulation index.

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to retrieve slider from
        """
        minimum = self.getMinimum()
        maximum = self.getMaximum()
        resolution = 1

        return sg.Slider(
            range=(minimum, maximum),
            default_value=minimum,
            resolution=resolution,
            orientation="horizontal",
            enable_events=True,
            size=self.getDimensions(name="parameter_slider_slider"),
            border_width=0,
            pad=(0, 0),
            key="-GROUP_INDEX SLIDER-"
        )

    def getMaximumLabel(self) -> sg.Text:
        """
        Get lable indicating maximum simulation index for slider.

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to retrieve label from
        """
        return self.maximum_label


class SampleIndexSlider(Slider):
    def __init__(
        self,
        window: Window,
        sample_count: int
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.SampleIndexSlider`.

        :param window: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` that slider is stored in
        :param sample_count: number of sample groups within sample size
        """
        assert isinstance(window, Window)
        Slider.__init__(
            self,
            window=window,
            minimum=1,
            maximum=sample_count,
            stepcount=sample_count
        )

        assert isinstance(sample_count, int)
        self.sample_count = sample_count

        slider = self.getSlider()
        name_label = sg.Text("Sample Group:")
        minimum_label = sg.Text("1")
        maximum_label = sg.Text(f"{sample_count:d}")
        self.maximum_label = maximum_label

        elements = [
            name_label,
            minimum_label,
            slider,
            maximum_label
        ]
        self.addElements(elements)

    def getSampleCount(self) -> int:
        """
        Get sample count for slider.

        :param self: :class:`~Layout.OptimizationSimulationWindow.SampleIndexSlider` to retrieve count from
        """
        return self.sample_count

    def updateSampleCount(self, sample_count: int) -> None:
        """
        Change sample count for slider (e.g. if user selected new sample size).

        :param self: :class:`~Layout.OptimizationSimulationWindow.SampleIndexSlider` to change count in
        :param sample_size: new number of samples for slider
        """
        assert isinstance(sample_count, int)

        maximum_label = self.getMaximumLabel()
        maximum_label.update(sample_count)

        slider = self.getSlider()
        minimum = self.getMinimum()
        slider.update(range=(minimum, sample_count))

        self.sample_count = sample_count

    @storeElement
    def getSlider(self) -> sg.Slider:
        """
        Get slider to take user input for simulation index.

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to retrieve slider from
        """
        minimum = self.getMinimum()
        maximum = self.getMaximum()
        resolution = self.getResolution()

        return sg.Slider(
            range=(minimum, maximum),
            default_value=minimum,
            resolution=resolution,
            orientation="horizontal",
            enable_events=True,
            size=self.getDimensions(name="parameter_slider_slider"),
            border_width=0,
            pad=(0, 0),
            key="-SAMPLE_INDEX SLIDER-"
        )

    def getMaximumLabel(self) -> sg.Text:
        """
        Get lable indicating maximum index for slider.

        :param self: :class:`~Layout.OptimizationSimulationWindow.GroupIndexSlider` to retrieve label from
        """
        return self.maximum_label


class SampleSizeRadioGroup(RadioGroup):
    def __init__(
        self,
        window: Window,
        sample_sizes: Tuple[int, ...]
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.SampleSizeRadioGroup`.

        :param axis_name: name of axis associated with group
        :param window: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` that group is stored in
        :param sample_sizes: collection of sample sizes from optimization fits
        """
        assert isinstance(window, Window)
        group_id = "SAMPLE SIZE"
        RadioGroup.__init__(
            self,
            radios=[],
            name=group_id,
            group_id=group_id,
            window=window
        )

        assert isinstance(sample_sizes, tuple)
        assert len(sample_sizes) >= 1
        for sample_size in sample_sizes:
            assert isinstance(sample_size, int)
        self.sample_sizes = sample_sizes

        for sample_size in sample_sizes:
            assert isinstance(sample_size, int)
            radio = self.getSampleSizeRadio(sample_size)
            self.addElements(radio)

    def getSampleSizes(
        self,
        index: int = None
    ) -> Tuple[int, ...]:
        """
        Get samples sizes of optimization fits.

        :param self: :class:`~Layout.OptimizationSimulationWindow.SampleSizeRadioGroup` to retrieve sizes from
        :param index: index to retrieve sample size at.
            Defaults to returning tuple of all sample sizes.
        """
        sample_sizes = self.sample_sizes
        if index is not None:
            sample_sizes = sample_sizes[index]

        return sample_sizes

    @storeElement
    def getSampleSizeRadio(
        self,
        sample_size: int,
        default_size: int = None
    ) -> sg.Radio:
        """
        Get element to take user input for sample size.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantityFrame` to retrieve element from
        :param sample_size: sample size for individual radio item within group
        :param default_size: default sample size
        """
        assert isinstance(sample_size, int)
        if default_size is None:
            default_size = self.getSampleSizes(index=0)
        else:
            assert isinstance(default_size, int)
        is_default = sample_size == default_size

        radio_group_id = self.getGroupId()
        radio_key = f"-{radio_group_id:s} {sample_size:d}-"

        radio = sg.Radio(
            text=sample_size,
            tooltip="Choose sample size of results to retrieve",
            group_id=radio_group_id,
            default=is_default,
            enable_events=True,
            key=radio_key
        )
        return radio


class FreeParameterSlider(Slider):
    def __init__(self) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.FreeParameterSlider`.

        """


class ParameterSelectionSection(Element):
    def __init__(
        self,
        window: Window,
        sample_count_per_sample_sizes: Dict[int, int]
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection`

        :param window: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` that section is stored in
        :param sample_count_per_sample_sizes: dictionary of simulation counts.
            Key is integer of sample size.
            Value is integer of number of simulations within sample size.
        """
        assert isinstance(window, Window)
        Element.__init__(
            self,
            window=window
        )

        assert isinstance(sample_count_per_sample_sizes, dict)
        for sample_size, sample_count in sample_count_per_sample_sizes.items():
            assert isinstance(sample_size, int)
            assert isinstance(sample_count, int)
        self.simulation_count_per_sample_sizes = sample_count_per_sample_sizes

        sample_sizes = self.getSampleSizes()
        self.sample_size_radio_group = SampleSizeRadioGroup(
            window=window,
            sample_sizes=sample_sizes
        )

        default_sample_count = self.getSampleCountPerSampleSize(sample_size=1)
        self.sample_index_slider = SampleIndexSlider(
            window=window,
            sample_count=default_sample_count
        )

        default_sample_size = self.getSampleSizes(0)
        self.group_index_slider = GroupIndexSlider(
            window=window,
            sample_size=default_sample_size
        )

    def getSampleCountPerSampleSize(
        self,
        sample_size: int = None,
    ) -> Union[int, Dict[int, int]]:
        """
        Get simulation count for sample size.

        :param self: :class`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` to retrieve count from
        :param sample_size: size of sample to retrive simulation count of
        """
        if sample_size is None:
            return self.simulation_count_per_sample_sizes
        sample_sizes = self.getSampleSizes()
        assert isinstance(sample_size, int)
        assert sample_size in sample_sizes

        simulation_count_per_sample_sizes = self.simulation_count_per_sample_sizes
        simulation_count_per_sample_size = simulation_count_per_sample_sizes[sample_size]
        return simulation_count_per_sample_size

    def getSampleSizes(
        self,
        index: int = None
    ) -> Tuple[int, ...]:
        """
        Get samples sizes of optimization fits.

        :param self: :class:`~Layout.OptimizationSimulationWindow.SampleSizeRadioGroup` to retrieve sizes from
        :param index: index to retrieve sample size at.
            Defaults to returning tuple of all sample sizes.
        """
        simulation_count_per_sample_sizes = self.simulation_count_per_sample_sizes
        sample_sizes = tuple(list(simulation_count_per_sample_sizes.keys()))
        if index is not None:
            sample_sizes = sample_sizes[index]

        return sample_sizes

    def getSampleSizeGroup(self) -> SampleSizeRadioGroup:
        """
        Get element to take user input for envelope (e.g. "Amplitude").

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve element from
        """
        return self.sample_size_radio_group

    def getGroupIndexSlider(self) -> GroupIndexSlider:
        """
        Get element to take user input for simulation index.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve element from
        """
        return self.group_index_slider

    def getSampleIndexSlider(self) -> SampleIndexSlider:
        """
        Get element to take user input for sample index.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve element from
        """
        return self.sample_index_slider

    def getChosenSampleSize(self) -> int:
        """
        Get sample size chosen by user.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve size from
        """
        sample_size_radio_group = self.getSampleSizeGroup()
        chosen_sample_size = int(sample_size_radio_group.getChosenRadioLabel())
        return chosen_sample_size

    def getChosenGroupIndex(self) -> int:
        """
        Get index of sample group chosen by user.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve index from
        """
        group_index_slider = self.getGroupIndexSlider()
        chosen_groupo_index = int(group_index_slider.getSliderValue())
        return chosen_groupo_index

    def getChosenSampleIndex(self) -> int:
        """
        Get index of sample chosen by user.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve index from
        """
        sample_index_slider = self.getSampleIndexSlider()
        chosen_sample_index = int(sample_index_slider.getSliderValue())
        return chosen_sample_index

    def getParameterIndex(self) -> Tuple[int, int]:
        """
        Get index for set of parameters chosen by user.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve index from
        """
        chosen_sample_size = self.getChosenSampleSize()
        chosen_sample_index = self.getChosenSampleIndex() - 1
        chosen_group_index = self.getChosenGroupIndex() - 1

        chosen_index = (chosen_sample_size, chosen_sample_index, chosen_group_index)
        return chosen_index

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout allowing user to select parameter index.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to retrieve layout from
        """
        layout_obj = Layout()

        sample_size_radio_group = self.getSampleSizeGroup()
        sample_size_row_elements = [
            sg.Text("Sample Size:"),
            *sample_size_radio_group.getRadios()
        ]
        sample_size_row = Row(elements=sample_size_row_elements)
        layout_obj.addRows(sample_size_row)

        sample_index_slider = self.getSampleIndexSlider()
        layout_obj.addRows(sample_index_slider)

        group_index_slider = self.getGroupIndexSlider()
        layout_obj.addRows(group_index_slider)

        layout = layout_obj.getLayout()
        return layout

    def changedSampleSizeGroup(self) -> None:
        """
        Perform action after user changes selection in sample size radio group.

        :param self: :class:`~Layout.OptimizationSimulationWindow.ParameterSelectionSection` to perform actions in
        """
        chosen_sample_size = self.getChosenSampleSize()
        new_sample_count = self.getSampleCountPerSampleSize(chosen_sample_size)

        sample_index_slider = self.getSampleIndexSlider()
        sample_index_slider.updateSampleCount(new_sample_count)

        group_index_slider = self.getGroupIndexSlider()
        group_index_slider.updateSimulationCount(chosen_sample_size)
