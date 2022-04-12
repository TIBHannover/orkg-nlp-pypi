import os
from huggingface_hub import hf_hub_download
from argparse import ArgumentParser

from orkgnlp.util import io
from orkgnlp.util.exceptions import ValidationError

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CACHE_ROOT = os.path.join(os.path.expanduser('~'), 'orkgnlp_data')
CACHE_ROOT = os.getenv('ORKG_NLP_DATA_CACHE_ROOT', DEFAULT_CACHE_ROOT)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-m',
                        '--model_names',
                        nargs='+',
                        type=str,
                        required=True,
                        help='List of model names to be downloaded.'
                        )

    return parser.parse_args()


def repos_are_known(model_names, orkg_models):
    return set(model_names).issubset(orkg_models.keys())


def download(model_names):
    if isinstance(model_names, str):
        model_names = [model_names]

    orkg_models = io.read_json(os.path.join(CURRENT_DIR, 'huggingface_repos.json'))
    if not repos_are_known(model_names, orkg_models):
        raise ValidationError('Unknown model name(s) given {}. Please check the following known models: {}'
                              .format(model_names, list(orkg_models.keys())))

    for model_name in model_names:
        for repo in orkg_models[model_name]:
            for filename in repo['files']:
                hf_hub_download(
                    repo_id=repo['repo_id'],
                    filename=filename,
                    force_filename=filename,
                    cache_dir=os.path.join(CACHE_ROOT, model_name)
                )


def main():
    args = parse_args()
    download(model_names=args.model_names)


if __name__ == '__main__':
    main()
