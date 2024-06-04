# Copyright (C) 2019, QuantStack
# SPDX-License-Identifier: BSD-3-Clause
# Copied from mamba 1.5.2
# flake8: noqa

import json
import os
import sys
import tempfile
import urllib.parse
from collections import OrderedDict

import libmambapy as api
from boltons.setutils import IndexedSet
from conda.base.constants import ChannelPriority
from conda.base.context import context
from conda.common.serialize import json_dump
from conda.common.url import join_url, remove_auth, split_anaconda_token
from conda.core.index import _supplement_index_with_system, check_allowlist
from conda.core.link import PrefixSetup, UnlinkLinkTransaction
from conda.core.prefix_data import PrefixData
from conda.core.solve import diff_for_unlink_link_precs
from conda.gateways.connection.session import CondaHttpAuth
from conda.models.channel import Channel as CondaChannel
from conda.models.prefix_graph import PrefixGraph
from conda.models.records import PackageRecord

from conda_forge_feedstock_check_solvable.utils import timer


def get_index(
    channel_urls=(),
    prepend=True,
    platform=None,
    use_local=False,
    use_cache=False,
    unknown=None,
    prefix=None,
    repodata_fn="repodata.json",
):
    if isinstance(platform, str):
        platform = [platform, "noarch"]

    all_channels = []
    if use_local:
        all_channels.append("local")
    all_channels.extend(channel_urls)
    if prepend:
        all_channels.extend(context.channels)
    check_allowlist(all_channels)

    # Remove duplicates but retain order
    all_channels = list(OrderedDict.fromkeys(all_channels))

    dlist = api.DownloadTargetList()

    index = []

    def fixup_channel_spec(spec):
        at_count = spec.count("@")
        if at_count > 1:
            first_at = spec.find("@")
            spec = (
                spec[:first_at]
                + urllib.parse.quote(spec[first_at])
                + spec[first_at + 1 :]
            )
        if platform:
            spec = spec + "[" + ",".join(platform) + "]"
        return spec

    all_channels = list(map(fixup_channel_spec, all_channels))
    pkgs_dirs = api.MultiPackageCache(context.pkgs_dirs)
    api.create_cache_dir(str(pkgs_dirs.first_writable_path))

    for channel in api.get_channels(all_channels):
        for channel_platform, url in channel.platform_urls(with_credentials=True):
            full_url = CondaHttpAuth.add_binstar_token(url)

            sd = api.SubdirData(
                channel, channel_platform, full_url, pkgs_dirs, repodata_fn
            )

            needs_finalising = sd.download_and_check_targets(dlist)
            index.append(
                (
                    sd,
                    {
                        "platform": channel_platform,
                        "url": url,
                        "channel": channel,
                        "needs_finalising": needs_finalising,
                    },
                )
            )

    for sd, info in index:
        if info["needs_finalising"]:
            sd.finalize_checks()
        dlist.add(sd)

    is_downloaded = dlist.download(api.MAMBA_DOWNLOAD_FAILFAST)

    if not is_downloaded:
        raise RuntimeError("Error downloading repodata.")

    return index


def load_channels(
    pool,
    channels,
    repos,
    has_priority=None,
    prepend=True,
    platform=None,
    use_local=False,
    use_cache=True,
    repodata_fn="repodata.json",
):
    with timer("load_channels get_index"):
        index = get_index(
            channel_urls=channels,
            prepend=prepend,
            platform=platform,
            use_local=use_local,
            repodata_fn=repodata_fn,
            use_cache=use_cache,
        )

    with timer("load_channels create_repo"):
        if has_priority is None:
            has_priority = context.channel_priority in [
                ChannelPriority.STRICT,
                ChannelPriority.FLEXIBLE,
            ]

        subprio_index = len(index)
        if has_priority:
            # first, count unique channels
            n_channels = len({entry["channel"].canonical_name for _, entry in index})
            current_channel = index[0][1]["channel"].canonical_name
            channel_prio = n_channels

        for subdir, entry in index:
            # add priority here
            if has_priority:
                if entry["channel"].canonical_name != current_channel:
                    channel_prio -= 1
                    current_channel = entry["channel"].canonical_name
                priority = channel_prio
            else:
                priority = 0
            if has_priority:
                subpriority = 0
            else:
                subpriority = subprio_index
                subprio_index -= 1

            if not subdir.loaded() and entry["platform"] != "noarch":
                # ignore non-loaded subdir if channel is != noarch
                continue

            if context.verbosity != 0 and not context.json:
                print(
                    "Channel: {}, platform: {}, prio: {} : {}".format(
                        entry["channel"], entry["platform"], priority, subpriority
                    )
                )
                print("Cache path: ", subdir.cache_path())

            repo = subdir.create_repo(pool)
            repo.set_priority(priority, subpriority)
            repos.append(repo)

    return index
