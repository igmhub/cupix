import yaml

# Read yaml config files and create dictionaries
class InferenceConfig(object):
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
        self.post_config = self.data.get('posterior_config', {})
        self.mini_config = self.data.get('minimizer_config', {})
        self.samp_config = self.data.get('sampler_config', {})
        self.params_config = self.data.get('params_config', {})
        self.remove_nones()
        self.verbose_update()
        
    
    def print_all(self):
        " Print all parameters"
        names = ["Posterior config", "Minimizer config", "Sampler config", "Params config"]
        for i, category in enumerate([self.post_config, self.mini_config, self.samp_config, self.params_config]):
            print(names[i])
            print(category)
        

    def remove_nones(self):
        " Remove None from dictionaries "
        new_post_config = {k: v for k, v in self.post_config.items() if v is not None}
        new_mini_config = {k: v for k, v in self.mini_config.items() if v is not None}
        new_samp_config = {k: v for k, v in self.samp_config.items() if v is not None}
        for par in self.params_config:
            if self.params_config[par] is not None:
                new_par_config = {k: v for k, v in self.params_config[par].items() if v is not None}
                self.params_config[par] = new_par_config
        new_params_config = {par: self.params_config[par] for par in self.params_config if self.params_config[par] is not None}
        

        self.post_config = new_post_config
        self.mini_config = new_mini_config
        self.samp_config = new_samp_config
        self.params_config = new_params_config


    def verbose_update(self):
        if self.verbose_all is not None:
            if self.verbose_all:
                for category in [self.post_config, self.mini_config, self.samp_config]:
                    category['verbose'] = True
            else:
                for category in [self.post_config, self.mini_config, self.samp_config]:
                    category['verbose'] = False

