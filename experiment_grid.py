import multiprocessing
import time
from multiprocessing import Pool


class ExperimentGrid:
    """
    Experiment Grid class based on
    `Spinning up <https://spinningup.openai.com/en/latest/utils/run_utils.html#experimentgrid>`_
    used to run a function over multiple parameter values either sequentially or in parallel
    
    NOTE: This parallelizes the training for the agent. I did NOT write this. Credit goes to 
    Austin Harris (https://www.linkedin.com/in/austinsharris).
    """

    def __init__(self, func, print_kwargs=False):
        self.print_kwargs = print_kwargs
        self.keys = []
        self.values = []
        self.func = func

    def add(self, arg_name: str, val):
        if not isinstance(val, list):
            val = [val]
        self.keys.append(arg_name)
        self.values.append(val)

    def _variants(self, keys, values):
        """
        Recursively create list of dictionaries of the cross-product of all possible key-value pairs
        Returns: flat list of variants
        """
        if len(keys) == 1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], values[1:])

        variants = []
        for val in values[0]:
            for pre_v in pre_variants:
                v = dict()
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        """
        Generate list of variants and un-flatten sub dictionaries
        Returns:
             list of un-flattened variants ready to be passed into a function
        """
        flat_variants = self._variants(self.keys, self.values)

        def unflatten(variant):
            new_variant = dict()
            unflatten_set = set()
            for k, v in variant.items():
                if ":" in k:
                    spits = k.split(":")
                    k0 = spits[0]

                    if k0 not in new_variant:
                        new_variant[k0] = dict()

                    new_variant[k0][":".join(spits[1:])] = v
                    unflatten_set.add(k0)
                else:
                    new_variant[k] = v
            for k in unflatten_set:
                new_variant[k] = unflatten(new_variant[k])
            return new_variant

        variants = [unflatten(variant) for variant in flat_variants]
        return variants

    def _print_variant(self, variant, level=0):
        p_str = ""
        for k in variant:
            p_str += "\t" * level
            p_str += f"{k}: "
            if isinstance(variant[k], dict):
                p_str += "\n"
                p_str += self._print_variant(variant[k], level=level + 1)
            else:
                if hasattr(variant[k], "__name__"):
                    p_str += f"{variant[k].__name__}"
                else:
                    p_str += f"{variant[k]}"
            p_str += "\n"
        return p_str[:-1]

    def print_variant(self, variant):
        print("Using kwargs:")
        print(f"{self._print_variant(variant, level=1)}")

    def run(self):
        """
        Run all variants sequentially
        """
        variants = self.variants()
        for _, variant in enumerate(variants):
            self._run_variant(variant)

    def _run_variant(self, variant):
        """
        Runs a variant
        Args:
            variant (dict): Variant to run
        """
        num = self.variants().index(variant)
        start_time = time.time()
        print(f"Starting experiment {num+1} at " f"{time.strftime('%H:%M:%S')}")
        if self.print_kwargs:
            self.print_variant(variant)
        self.func(**variant)
        run_time = time.time() - start_time
        print(f"Finished exp {num+1} in {run_time:.3f}")

    def run_mult(self, num_procs=None):
        """
        Run all variants in parallel
        Args:
            num_procs: Maximum number of processes to run simultaneously.
                if None defaults to :func:`multiprocessing.cpu_count`
        """
        if num_procs is None:
            num_procs = multiprocessing.cpu_count()

        variants = self.variants()
        with Pool(processes=num_procs) as pool:
            pool.map(self._run_variant, variants)
            pool.close()
            pool.join()
