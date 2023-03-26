---
layout: page
title: People
permalink: /scientists/
---

{% assign people = site.scientists | where: "position", "Professor" %}
{% if people.size > 0 %}
# Professors
{% for scientist in people %}
[{{ scientist.title }}]({{ scientist.url | relative_url }})
{% endfor %}
{% endif %}

{% assign people = site.scientists | where: "position", "Lecturer" %}
{% if people.size > 0 %}
# Lecturers
{% for scientist in people %}
[{{ scientist.title }}]({{ scientist.url | relative_url }})
{% endfor %}
{% endif %}

{% assign people = site.scientists | where: "position", "Researcher" %}
{% if people.size > 0 %}
# Research Staff
{% for scientist in people %}
[{{ scientist.title }}]({{ scientist.url | relative_url }})
{% endfor %}
{% endif %}