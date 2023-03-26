---
layout: page
title: People
permalink: /people/
---

{% assign people = site.people | where: "position", "Professor" %}
{% if people.size > 0 %}
# Professors
{% for person in people %}
[{{ person.title }}]({{ person.url | relative_url }})
{% endfor %}
{% endif %}

{% assign people = site.people | where: "position", "Lecturer" %}
{% if people.size > 0 %}
# Lecturers
{% for person in people %}
[{{ person.title }}]({{ person.url | relative_url }})
{% endfor %}
{% endif %}

{% assign people = site.people | where: "position", "Researcher" %}
{% if people.size > 0 %}
# Research Staff
{% for person in people %}
[{{ person.title }}]({{ person.url | relative_url }})
{% endfor %}
{% endif %}