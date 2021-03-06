from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--num_test', type=int, default=256, help='how many test images to run')
        parser.add_argument('--re_index', action='store_true', help='Named of re-index for test image.')
        parser.add_argument('--result_name', type=str, default='brighstest_eval', help='Name of result csv')
        parser.add_argument('--no_save_image', action='store_true', help='Not to save result image.')
        parser.add_argument('--test_mode', type=int, default=0, help='0:All, 1:Eval, 2:IIW, 3:Saw')
        parser.add_argument('--test_start', type=int, default=0, help='test start index')
        parser.add_argument('--eval_mask_calc_bp', action='store_true', help='Are masks taken into account when calculating bp.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
