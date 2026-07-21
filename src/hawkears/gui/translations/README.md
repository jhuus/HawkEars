# GUI translations

HawkEars currently ships with its English source text only. To add a language:

1. Run `hatch run hawkears:translations <target.ts>` to create or update a Qt
   source catalog, using a target such as
   `src/hawkears/gui/translations/hawkears_fr.ts`.
2. Translate the resulting `.ts` file with Qt Linguist.
3. Run `hatch run hawkears:translations-compile` to create the packaged `.qm`
   file.
4. Store the locale code under the `language` key in `QSettings`. A future
   language selector will manage this setting and prompt for an application
   restart.

Interface translation is intentionally separate from localized species and
administrative-area names. Those require catalogs keyed by stable identifiers.
