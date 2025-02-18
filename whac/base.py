import abc
import torch.optim
from lib.utils.timer import Timer
from lib.utils.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
import torch.utils.data.distributed
from whac.modules import get_model_whac

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='whac_logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Demoer_WHAC(Base):
    def __init__(self, config):
        super(Demoer_WHAC, self).__init__(config, log_name='test_logs.txt')
        self.cfg = config

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model_whac('test', self.cfg)
        model = DataParallel(model).cuda()

        self.logger.info('Load checkpoint from {}'.format(self.cfg.model.pretrained_model_path))
        ckpt = torch.load(self.cfg.model.pretrained_model_path, map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            if 'module' not in k:
                k = 'module.' + k
            new_state_dict[k] = v
        self.logger.warning("Attention: Strict=False is set for checkpoint loading. Please check manually.")
        model.load_state_dict(new_state_dict, strict=True)
        model.cuda()
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)