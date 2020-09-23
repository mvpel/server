#!/usr/bin/env python3
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import docker
import os.path
import multiprocessing
import pathlib
import shutil
import subprocess
import sys
import traceback
from distutils.dir_util import copy_tree

#
# Build Triton Inference Server.
#
EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
FLAGS = None
LOG_FILE = None

# Map from container version to corresponding component versions
# container-version -> (ort version, ort openvino version)
CONTAINER_VERSION_MAP = {
    '20.08': ('1.4.0', '2020.2'),
    '20.09': ('1.4.0', '2020.2')
}


def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg)
        except Exception:
            print('<failed to log>')
    if LOG_FILE is not None:
        try:
            LOG_FILE.write(msg)
        except Exception:
            LOG_FILE.write('<failed to log>')


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    fail_if(True, msg)


def fail_if(p, msg):
    if p:
        if LOG_FILE is not None:
            LOG_FILE.write('error: {}'.format(msg))
        print('error: {}'.format(msg))
        sys.exit(1)


def mkdir(path):
    log_verbose('mkdir: {}'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    log_verbose('rmdir: {}'.format(path))
    shutil.rmtree(path, ignore_errors=True)


def cpdir(src, dest):
    log_verbose('cpdir: {} -> {}'.format(src, dest))
    copy_tree(src, dest, preserve_symlinks=1)


def gitclone(cwd, repo, tag):
    log_verbose('git clone of repo "{}" at tag "{}"'.format(repo, tag))
    p = subprocess.Popen([
        'git', 'clone', '--single-branch', '--depth=1', '-b', tag,
        'https://github.com/triton-inference-server/{}.git'.format(repo)
    ],
                         cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0,
            'git clone of repo "{}" at tag "{}" failed'.format(repo, tag))


def cmake(cwd, args):
    log_verbose('cmake {}'.format(args))
    p = subprocess.Popen([
        'cmake',
    ] + args, cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'cmake failed')


def makeinstall(cwd):
    log_verbose('make install')
    verbose_flag = 'VERBOSE=1' if FLAGS.verbose else ''
    p = subprocess.Popen([
        'make', 'install', '-j',
        str(FLAGS.build_parallel), verbose_flag, 'install'
    ],
                         cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'make install failed')


def backend_repo(be):
    if (be == 'tensorflow1') or (be == 'tensorflow2'):
        return 'tensorflow_backend'
    return '{}_backend'.format(be)


def backend_cmake_args(components, be, install_dir):
    if be == 'onnxruntime':
        args = onnxruntime_cmake_args()
    elif be == 'tensorflow1':
        args = tensorflow_cmake_args(1)
    elif be == 'tensorflow2':
        args = tensorflow_cmake_args(2)
    elif be == 'python':
        args = []
    elif be == 'dali':
        args = dali_cmake_args()
    elif be in EXAMPLE_BACKENDS:
        args = []
    else:
        fail('unknown backend {}'.format(be))

    cargs = args + [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    if FLAGS.disable_gpu:
        cargs.append('-DTRITON_ENABLE_GPU:BOOL=OFF')

    cargs.append('..')
    return cargs


def onnxruntime_cmake_args():
    return [
        '-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON',
        '-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON',
        '-DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/opt/tritonserver/include/onnxruntime',
        '-DTRITON_ONNXRUNTIME_LIB_PATHS=/opt/tritonserver/backends/onnxruntime'
    ]


def tensorflow_cmake_args(ver):
    image = 'nvcr.io/nvidia/tensorflow:{}-tf{}-py3'.format(
        FLAGS.upstream_container_version, ver)
    return [
        '-DTRITON_TENSORFLOW_VERSION={}'.format(ver),
        '-DTRITON_TENSORFLOW_DOCKER_IMAGE={}'.format(image)
    ]


def dali_cmake_args():
    return [
        '-DTRITON_DALI_SKIP_DOWNLOAD=OFF',
    ]


def container_build(container_version):
    log('Create build container')

    # Set the docker build-args based on 'container_version'
    if container_version in CONTAINER_VERSION_MAP:
        onnx_runtime_version = CONTAINER_VERSION_MAP[container_version][0]
        onnx_runtime_openvino_version = CONTAINER_VERSION_MAP[
            container_version][1]
    else:
        fail('unsupported container version {}'.format(container_version))

    client = docker.from_env()

    buildargs = {
        'TRITON_VERSION':
            FLAGS.version,
        'TRITON_CONTAINER_VERSION':
            container_version,
        'BASE_IMAGE':
            'nvcr.io/nvidia/tritonserver:{}-py3'.format(container_version),
        'PYTORCH_IMAGE':
            'nvcr.io/nvidia/pytorch:{}-py3'.format(container_version),
        'ONNX_RUNTIME_VERSION':
            onnx_runtime_version,
        'ONNX_RUNTIME_OPENVINO_VERSION':
            onnx_runtime_openvino_version
    }

    log_verbose('container build {}'.format(buildargs))
    try:
        # First build Dockerfile.build. Because of the way Docker does
        # caching with multi-stage images, we must build each stage
        # separately to make sure it is cached (specifically this is
        # needed for CI builds where the build starts with a clean
        # docker cache each time).

        # PyTorch
        image, output = client.images.build(path='.',
                                            rm=True,
                                            quiet=False,
                                            pull=True,
                                            tag='tritonserver_pytorch',
                                            dockerfile='Dockerfile.build',
                                            target='tritonserver_pytorch',
                                            cache_from=[
                                                'tritonserver_pytorch_cache0',
                                                'tritonserver_pytorch_cache1'
                                            ],
                                            buildargs=buildargs)
        if FLAGS.verbose:
            for ln in output:
                log_verbose(ln)

        # ONNX Runtime
        image, output = client.images.build(path='.',
                                            rm=True,
                                            quiet=False,
                                            pull=True,
                                            tag='tritonserver_onnx',
                                            dockerfile='Dockerfile.build',
                                            target='tritonserver_onnx',
                                            cache_from=[
                                                'tritonserver_pytorch_cache0',
                                                'tritonserver_pytorch_cache1'
                                                'tritonserver_onnx_cache0',
                                                'tritonserver_onnx_cache1'
                                            ],
                                            buildargs=buildargs)
        if FLAGS.verbose:
            for ln in output:
                log_verbose(ln)

        # Final build image
        image, output = client.images.build(path='.',
                                            rm=True,
                                            quiet=False,
                                            pull=True,
                                            tag='tritonserver_build',
                                            dockerfile='Dockerfile.build',
                                            cache_from=[
                                                'tritonserver_pytorch_cache0',
                                                'tritonserver_pytorch_cache1'
                                                'tritonserver_onnx_cache0',
                                                'tritonserver_onnx_cache1'
                                                'tritonserver_build_cache0',
                                                'tritonserver_build_cache1'
                                            ],
                                            buildargs=buildargs)
        if FLAGS.verbose:
            for ln in output:
                log_verbose(ln)

        # Next run build.py inside the container with the same flags
        # as was used to run this instance, except with
        # --container-version changes to --upstream-container-version.
        runargs = [
            a.replace('--container-version', '--upstream-container-version')
            for a in sys.argv
        ]
        container = client.containers.run(
            image,
            'build.py {}'.format(runargs.join(' ')),
            detach=True,
            name='tritonserver_builder0',
            volumes={
                '/var/run/docker.sock': {
                    'bind': '/var/run/docker.sock',
                    'mode': 'rw'
                }
            })
        if FLAGS.verbose:
            for ln in container.logs(stream=True):
                log_verbose(ln)

    except Exception as e:
        logging.error(traceback.format_exc())
        fail('container build failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Used internally for docker build, not intended for direct use
    parser.add_argument('--upstream-container-version',
                        type=str,
                        required=False,
                        help=argparse.SUPPRESS)

    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument('-q',
                          '--quiet',
                          action="store_true",
                          required=False,
                          help='Disable console output.')
    group_qv.add_argument('-v',
                          '--verbose',
                          action="store_true",
                          required=False,
                          help='Enable verbose output.')
    parser.add_argument(
        '--log-file',
        type=str,
        required=False,
        help=
        'File where verbose output should be recorded, even if --quiet flag is specified.'
    )

    parser.add_argument(
        '--build-dir',
        type=str,
        required=True,
        help=
        'Build directory. All repo clones and builds will be performed in this directory.'
    )
    parser.add_argument(
        '--install-dir',
        required=False,
        default=None,
        help='Install directory, default is <builddir>/opt/tritonserver.')
    parser.add_argument(
        '--build-type',
        required=False,
        default='Release',
        help=
        'Build type, one of "Release", "Debug", "RelWithDebInfo" or "MinSizeRel". Default is "Release".'
    )
    parser.add_argument(
        '-j',
        '--build-parallel',
        type=int,
        required=False,
        default=None,
        help='Build parallelism. Defaults to 2 * number-of-cores.')

    parser.add_argument('--version',
                        type=str,
                        required=False,
                        default='0.0.0',
                        help='The Triton version.')
    parser.add_argument(
        '--container-version',
        type=str,
        required=False,
        help=
        'The Triton container version. If specified, Docker will be used for the build and component versions will be set automatically.'
    )

    parser.add_argument('--disable-gpu',
                        action="store_true",
                        required=False,
                        help='Disable GPU support.')

    parser.add_argument(
        '-b',
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>[:<repo-tag>]. <repo-tag> indicates the git tag/branch to use for the build, default is "main".'
    )
    parser.add_argument(
        '-r',
        '--repo-tag',
        action='append',
        required=False,
        help=
        'The git tag/branch to use for a component of the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", or "backend".'
    )

    FLAGS = parser.parse_args()

    if FLAGS.log_file is not None:
        LOG_FILE = open(FLAGS.log_file, "w")

    # If --container-version is specified then we use Dockerfile.build
    # to create the appropriate base build container and then perform
    # the actual build within that container.
    if FLAGS.container_version is not None:
        container_build(FLAGS.container_version)
        sys.exit(0)

    log('Building Triton Inference Server')

    if FLAGS.install_dir is None:
        FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")
    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2
    if FLAGS.version is None:
        FLAGS.version = '0.0.0'

    # Initialize map of common components and repo-tag for each.
    components = {'common': 'main', 'core': 'main', 'backend': 'main'}
    if FLAGS.repo_tag:
        for be in FLAGS.repo_tag:
            parts = be.split(':')
            fail_if(
                len(parts) != 2,
                '--repo-tag must specific <component-name>:<repo-tag>')
            fail_if(
                parts[0] not in components,
                '--repo-tag <component-name> must be "common", "core", or "backend"'
            )
            components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    if FLAGS.backend:
        for be in FLAGS.backend:
            parts = be.split(':')
            if len(parts) == 1:
                parts.append('main')
            log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
            backends[parts[0]] = parts[1]

    # Build each backend...
    for be in backends:
        repo = backend_repo(be)
        repo_build_dir = os.path.join(FLAGS.build_dir, repo, 'build')
        repo_install_dir = os.path.join(repo_build_dir, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, repo, backends[be])
        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              backend_cmake_args(components, be, repo_install_dir))
        makeinstall(repo_build_dir)

        backend_install_dir = os.path.join(FLAGS.install_dir, 'backends', be)
        rmdir(backend_install_dir)
        mkdir(backend_install_dir)
        cpdir(os.path.join(repo_install_dir, 'backends', be),
              backend_install_dir)
