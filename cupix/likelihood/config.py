import yaml
from cupix.likelihood import theory, model_lya, model_contaminants

# Read yaml config files and create dictionaries
class Config(object):
    " Config class, reads yaml files and creates dictionaries "
    def __init__(
        self,
        yaml_file
    ):
        with open(yaml_file) as stream:
            try:
                self.data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        self.verbose_all = self.data.get('verbose_all', None)
        self.cosmo_config = self.data.get('cosmo_config', {})
        self.theory_config = self.data.get('theory_config', {})
        self.like_config = self.data.get('likelihood_config', {})
        self.post_config = self.data.get('posterior_config', {})
        self.mini_config = self.data.get('minimizer_config', {})
        self.samp_config = self.data.get('sampler_config', {})
        self.data_config = self.data.get('data_config', {})

        # check theory params
        model_lya.no_unrecognized_igm_params(self.theory_config.get('igm_config', {}))
        model_lya.no_unrecognized_lya_params(self.theory_config.get('lya_config', {}))
        model_contaminants.no_unrecognized_cont_params(self.theory_config.get('contaminant_config', {}))
        
        # make one large theory config without sub-dictionaries
        self.consolidate_theory_config()
        self.remove_nones()
        # make sure theory config is self-consistent
        model_lya.no_conflicting_params(self.theory_config)
        self.verbose_update()
        
    def consolidate_theory_config(self):
        " Make one large theory config without sub-dictionaries, for easy use in theory.py."
        consolidated_config = {}
        sub_config_names = ['lya_config', 'igm_config', 'contaminant_config']
        for param in self.theory_config:
            if param not in sub_config_names:
                consolidated_config[param] = self.theory_config[param]
        # add the rest of the theory config parameters that are in sub-dictionaries
        for sub_config_name in sub_config_names:
            sub_config = self.theory_config.get(sub_config_name, {})
            for param in sub_config:
                if param in consolidated_config:
                    raise ValueError(f"Duplicate parameter {param} found in theory config.")
                consolidated_config[param] = sub_config[param]
        self.theory_config = consolidated_config

    def print_all(self):
        " Print all parameters"
        names = ["Cosmo config", "Theory config", "Likelihood config", "Posterior config", "Minimizer config", "Sampler config", "Data config"]
        for i, category in enumerate([self.cosmo_config, self.theory_config, self.like_config, self.post_config, self.mini_config, self.samp_config, self.data_config]):
            print(names[i])
            print(category)
        

    def remove_nones(self):
        " Remove None from dictionaries "
        new_theory_config = {k: v for k, v in self.theory_config.items() if v is not None}
        new_like_config = {k: v for k, v in self.like_config.items() if v is not None}
        new_post_config = {k: v for k, v in self.post_config.items() if v is not None}
        new_mini_config = {k: v for k, v in self.mini_config.items() if v is not None}
        new_samp_config = {k: v for k, v in self.samp_config.items() if v is not None}
        new_data_config = {k: v for k, v in self.data_config.items() if v is not None}
        self.theory_config = new_theory_config
        self.like_config = new_like_config
        self.post_config = new_post_config
        self.mini_config = new_mini_config
        self.samp_config = new_samp_config
        self.data_config = new_data_config


    def verbose_update(self):
        if self.verbose_all is not None:
            if self.verbose_all:
                for category in [self.theory_config, self.like_config, self.post_config, self.mini_config, self.samp_config, self.data_config]:
                    category['verbose'] = True
            else:
                for category in [self.theory_config, self.like_config, self.post_config, self.mini_config, self.samp_config, self.data_config]:
                    category['verbose'] = False