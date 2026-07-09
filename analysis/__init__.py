"""Analysis package: one module per analytical concern.

Modules follow a common pattern:

* ``extract_*`` functions operate on the raw Match-V5 timeline (via
  :class:`analysis.timeline.TimelineContext`) and are called by the parser.
* aggregate/dataframe functions operate on parsed
  :class:`models.MatchRecord` collections and are called by the pipeline.
"""
