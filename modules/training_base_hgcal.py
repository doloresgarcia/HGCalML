from DeepJetCore.training.training_base import training_base
from argparse import ArgumentParser


class HGCalTraining(training_base):
    def __init__(self, *args, parser=None, **kwargs):
        """
        Adds file logging
        """
        # use the DJC training base option to pass a parser
        if parser is None:
            parser = ArgumentParser("Run the training")
        parser.add_argument(
            "--interactive",
            help="prints output to screen",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--pretrained",
            "-p",
            help="Path to pretrained model checkpoint",
            default="",
            type=str,
        )

        # no reason for a lot of validation samples usually
        super().__init__(
            *args, resumeSilently=True, parser=parser, splittrainandtest=0.95, **kwargs
        )

        if not self.args.interactive:
            print(
                ">>> redirecting the following stdout and stderr to logs in",
                self.outputDir,
            )
            import sys

            sys.stdout = open(self.outputDir + "/stdout.txt", "w")
            sys.stderr = open(self.outputDir + "/stderr.txt", "w")

        from config_saver import copyModules

        copyModules(self.outputDir)  # save the modules with indexing for overwrites

    def compileModel(self, **kwargs):
        super().compileModel(is_eager=True, loss=None, **kwargs)
        if self.args.pretrained != "":
            print("*** Loading pretrained model from", self.args.pretrained)
            self.keras_model.load_weights(self.args.pretrained)

    def trainModel(
        self, nepochs, batchsize, backup_after_batches=100, checkperiod=1, **kwargs
    ):
        """
        Just implements some defaults
        """
        return super().trainModel(
            nepochs=nepochs,
            batchsize=batchsize,
            run_eagerly=True,
            verbose=2,
            batchsize_use_sum_of_squares=False,
            fake_truth=True,
            backup_after_batches=backup_after_batches,
            checkperiod=checkperiod,
            **kwargs
        )
