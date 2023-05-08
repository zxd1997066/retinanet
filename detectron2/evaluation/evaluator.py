# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

from detectron2.utils.comm import is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


def inference_on_dataset(cfg, model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = cfg.num_warmup
    num_iters = cfg.num_iters
    start_time = time.perf_counter()
    total_compute_time = 0
    sum_time = 0.0
    sum_sample = 0
    batch_time_list =[]
    with inference_context(model), torch.no_grad():
        model.eval()
        if cfg.ipex:
            print("---- Running with IPEX")
            import intel_extension_for_pytorch as ipex
            if cfg.precision == "bfloat16":
                model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            else:
                model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        for idx, inputs in enumerate(data_loader):
            if num_iters != 0 and idx > num_iters:
                break
            if cfg.jit and idx == 0:
                try:
                    model = torch.jit.trace(model, inputs, check_trace=False)
                    print("---- Use trace model.")
                except:
                    model = torch.jit.script(model)
                    print("---- Use script model.")
                if cfg.ipex:
                    model = torch.jit.freeze(model)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            tic = time.time()
            if cfg.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                    outputs = model(inputs)
                if idx == int(num_iters/2):
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        os.makedirs(timeline_dir)
                    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                "retinanet" + str(idx) + '-' + str(os.getpid()) + '.json'
                    print(timeline_file)
                    prof.export_chrome_trace(timeline_file)
                    table_res = prof.key_averages().table(sort_by="cpu_time_total")
                    print(table_res)
                    # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
            else:
                outputs = model(inputs)
            toc = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            print("Iteration: {}, inference time: {} sec.".format(idx, toc - tic), flush=True)
            if idx >= num_warmup:
                batch_time_list.append((toc - tic) * 1000)
                sum_time += (toc - tic)
                sum_sample += 1
            if cfg.eval_only:
                continue
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
    if cfg.eval_only:
        print("\n", "-"*20, "Summary", "-"*20)
        latency = sum_time / sum_sample * 1000
        throughput = sum_sample / sum_time
        print("inference latency:\t {:.3f} ms".format(latency))
        print("inference Throughput:\t {:.2f} samples/s".format(throughput))
        # P50
        batch_time_list.sort()
        p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
        p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
        p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
        print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                % (p50_latency, p90_latency, p99_latency))

        exit()
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep

    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    logger.info(
        "inference Throughput: {:.2f} imgs/s, on {} devices)".format(
            (total - num_warmup) / total_compute_time, num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
