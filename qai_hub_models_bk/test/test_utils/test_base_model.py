# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Any

from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)


class SimpleBaseModel(BaseModel):
    def get_input_spec(*args: Any, **kwargs: Any) -> None:
        return None

    def get_output_names(*args: Any, **kwargs: Any) -> None:
        return None

    @classmethod
    def from_pretrained(cls) -> "SimpleBaseModel":
        return cls()


def test_collection_model_demo() -> None:
    """Demo on how to use CollectionModel"""

    class Component1(SimpleBaseModel):
        pass

    class Component2(SimpleBaseModel):
        pass

    class Component3(SimpleBaseModel):
        pass

    @CollectionModel.add_component(Component1, "component_1")
    @CollectionModel.add_component(Component2, "component_2")
    class DummyCollection(PretrainedCollectionModel):
        @classmethod
        def from_pretrained(cls) -> "DummyCollection":
            return cls(*[c.from_pretrained() for c in cls.component_classes.values()])

    # Second subclass shouldn't interfere with DummyCollection
    @CollectionModel.add_component(Component1, "component_1")
    @CollectionModel.add_component(Component2, "component_2")
    @CollectionModel.add_component(Component3, "component_3")
    class SecondCollection(PretrainedCollectionModel):
        pass

    # Second subclass shouldn't interfere with DummyCollection
    @CollectionModel.add_component(Component1, "component_1")
    @CollectionModel.reset_components()
    class ThirdCollection(SecondCollection):
        pass

    # Access class vars via component_class_names
    assert DummyCollection.component_class_names == ["component_1", "component_2"]
    assert ThirdCollection.component_class_names == ["component_1"]

    model = DummyCollection.from_pretrained()

    # Access components via model (instance-level access works with generics)
    assert model.component_classes["component_1"] is Component1
    assert model.component_classes["component_2"] is Component2
    assert list(model.components.keys()) == ["component_1", "component_2"]
    assert isinstance(model.components["component_1"], Component1)

    # Instantiate with __init__ directly with positional args
    comp1_instance = Component1()
    comp2_instance = Component2()
    model3 = DummyCollection(comp1_instance, comp2_instance)
    assert model3.components["component_1"] is comp1_instance
    assert model3.components["component_2"] is comp2_instance
