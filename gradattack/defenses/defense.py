from gradattack.trainingpipeline import TrainingPipeline


class GradientDefense:
    """Applies a gradient defense to a given pipeline.
    **WARNING** This may modify the pipeline via monkey-patching of defenses!
    Please use with care."""
    def apply(self, pipeline: TrainingPipeline):
        assert (self.defense_name not in pipeline.applied_defenses
                ), f"Tried to apply duplicate defense {self.defense_name}!"
        pipeline.applied_defenses.append(self.defense_name)

    @property
    def defense_name(self):
        return self.__class__.__name__
