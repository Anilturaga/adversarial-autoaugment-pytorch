from untangle.external import augment_api
from torch.cuda import is_available
import os
class UntangleAI(object):
    def __init__(self):
        """untangle_license = UntangleLicense()
        status, msg = untangle_license.activate_license()
        if not status:
            print(msg)
            sys.exit(-1)
"""
        self.version = '2.0.0'
        self.authors = 'UntangleAI PTE LTD'
    def check_update_compatibility(self,tool,args):
        if tool['name'] == 'signal_estimation':
            module_path = os.path.abspath(os.getcwd())
            proj_path = os.path.abspath(os.path.join(module_path, args.mname))
            signal_path = os.path.abspath(os.path.join(proj_path, 'signal_estimation'))
            model_signal_data_path = os.path.join(signal_path, 'model_signal_data/')
            signal_store_path = os.path.join(model_signal_data_path, '{}_signals'.format(args.mname))
            results_path = os.path.join(signal_path, 'results')
            create_dirs = [proj_path,signal_path,model_signal_data_path,results_path]
            if tool['train'] == True:                
                #self.signal_estimation_results = os.path.join(self.signal_path, 'results/')
                for dirs in create_dirs:
                    if(not os.path.exists(dirs)):
                        os.makedirs(dirs)
                #print("New project "+args.mname+" created!")
                return signal_store_path
            else:
                out_prefix_idx = os.path.join(results_path, tool['experiment_ID'])
                out_prefix = os.path.join(out_prefix_idx, '{}_signals'.format(tool['experiment_ID']))
                if(not os.path.exists(model_signal_data_path)):
                    raise Exception("Patterns not found. Please use estimate_signals first with same mname to create patterns!")
                if (os.path.exists(out_prefix)):
                    raise Exception("Experiment with this ID already exists. Please use a new ID or delete this experiment using delete_signal_experiment fucntion")
                else:
                    os.makedirs(out_prefix_idx)
                return [signal_store_path,out_prefix]
        elif tool['name'] == 'uncertainity_tool':
            module_path = os.path.abspath(os.getcwd())
            proj_path = os.path.abspath(os.path.join(module_path, args.mname))
            uncertainity_path = os.path.abspath(os.path.join(proj_path, 'uncertainity_tool'))
            model_uncrt_data_path = os.path.join(uncertainity_path, 'model_uncrt_data/')
            uncrt_store_path = os.path.join(model_uncrt_data_path, '{}_uncertainty'.format(args.mname))
            results_path = os.path.join(uncertainity_path, 'results')
            create_dirs = [proj_path,uncertainity_path,model_uncrt_data_path,results_path]
            if tool['train'] == True:                
                #self.signal_estimation_results = os.path.join(self.signal_path, 'results/')
                for dirs in create_dirs:
                    if(not os.path.exists(dirs)):
                        os.makedirs(dirs)
                #print("New project "+args.mname+" created!")
                return uncrt_store_path
            else:
                out_path = os.path.join(results_path, tool['experiment_ID'])                
                if(not os.path.exists(model_uncrt_data_path)):
                    raise Exception("Uncertainity stats not found. Please use model_uncertainty first with same mname to create patterns!")
                if (os.path.exists(out_path)):
                    raise Exception("Experiment with this ID already exists. Please use a new ID or delete this experiment using delete_signal_experiment fucntion")
                else:
                    os.makedirs(out_path)
                return [uncrt_store_path,out_path]
        elif tool['name'] == 'concept_extraction':
            module_path = os.path.abspath(os.getcwd())
            proj_path = os.path.abspath(os.path.join(module_path, args.mname))
            concept_extraction_path = os.path.abspath(os.path.join(proj_path, 'concept_extraction'))
            create_dirs = [proj_path,concept_extraction_path]
            for dirs in create_dirs:
                if(not os.path.exists(dirs)):
                    os.makedirs(dirs)
            exp_path = os.path.abspath(os.path.join(concept_extraction_path, tool['experiment_ID']))
            if (os.path.exists(exp_path)):
                raise Exception("Experiment with this ID already exists. Please use a new ID or delete this experiment using delete_signal_experiment fucntion")
            else:
                os.makedirs(exp_path)
            return exp_path
        else:
            module_path = os.path.abspath(os.getcwd())
            proj_path = os.path.abspath(os.path.join(module_path, args.mname))
            train_augment_path = os.path.abspath(os.path.join(proj_path, 'train_augment'))
            create_dirs = [proj_path,train_augment_path]
            for dirs in create_dirs:
                if(not os.path.exists(dirs)):
                    os.makedirs(dirs)
            exp_path = os.path.abspath(os.path.join(train_augment_path, tool['experiment_ID']))
            if (os.path.exists(exp_path)):
                #raise Exception("Experiment with this ID already exists. Please use a new ID or delete this experiment using delete_signal_experiment fucntion")
                print()
            else:
                os.makedirs(exp_path)
            return exp_path


    def train_augment(self,model,optimizer,scheduler,train_dataset,valid_dataset,args,experiment_ID):
        if not is_available() or args.gpu_count == 0:
            raise Exception("Sorry, cuda not found.")
        tool={
            'name': 'train_augment',
            'experiment_ID': str(experiment_ID)
        }
        exp_path = self.check_update_compatibility(tool,args)
        return augment_api.train_augment(model,optimizer,scheduler,train_dataset,valid_dataset,args,exp_path)
        
