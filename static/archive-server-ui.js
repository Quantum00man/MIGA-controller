/* Optional adapter: the controller serves the same page without this script. */
(() => {
    const match = location.pathname.match(/^\/archive-server\/view\/([a-z0-9._-]+)\/archive\.html$/);
    if (!match) return;
    const device = match[1];
    const prefix = `/archive-server/view/${device}`;
    let app;
    axios.defaults.baseURL = prefix;
    axios.interceptors.request.use(request => {
        request.headers['X-Archive-Version'] = app?.serverVersion || 'latest';
        return request;
    });
    axios.interceptors.response.use(response => {
        const url = response.config.url || '';
        const saved = response.config.method !== 'get' && (
            /\/archive\/(overwrite|phase-reference-override|sync-differential-fit|sync-phase-calibration-optimization\/|sync-analysis-copies\/save)/.test(url)
            || response.data?.saved_intf_alpha_analysis_copy);
        if (saved && app) app.serverVersion = 'latest';
        return response;
    });
    const create = Vue.createApp;
    Vue.createApp = function(options, ...args) {
        const originalData = options.data;
        options.data = function() {
            return {...originalData.call(this), archiveServerMode: true, serverDeviceName: device,
                serverVersion: new URLSearchParams(location.search).get('server_version') || 'latest', serverVersions: [], serverVersionError: ''};
        };
        const methods = options.methods;
        const shareUrl = methods.loadedRunShareUrl;
        methods.loadedRunShareUrl = function(...values) {
            const original = shareUrl.apply(this, values);
            if (!original) return original;
            const url = new URL(original);
            const version = this.serverVersion === 'latest' ? this.serverVersions[0]?.id || 'original' : this.serverVersion;
            url.searchParams.set('server_version', version);
            return url.toString();
        };
        methods.serverArchiveUrl = function(url) {
            if (!url?.startsWith('/archive/')) return url;
            return prefix + url + (url.includes('?') ? '&' : '?') + 'server_version=' + encodeURIComponent(this.serverVersion);
        };
        for (const name of ['archiveAuditReportUrl', 'braggFringeCalibrationMotUrl', 'optimizationArtifactUrl']) {
            const original = methods[name];
            methods[name] = function(...values) { return this.serverArchiveUrl(original.apply(this, values)); };
        }
        methods.refreshServerVersions = async function() {
            const ref = this.loadedRunRef;
            if (!ref) return;
            try {
                const result = await axios.get(`/archive/versions/${ref.year}/${ref.month}/${ref.day}/${ref.run_id}`);
                this.serverVersions = result.data.versions;
                this.serverVersionError = '';
            } catch (error) { this.serverVersionError = error.response?.data?.detail || error.message; }
        };
        const load = methods.loadRun;
        methods.loadRun = async function(...values) {
            const key = [this.selectedYear, this.selectedMonth, this.selectedDay, this.selectedRun].join('/');
            if (this._serverReferenceKey && key !== this._serverReferenceKey) this.serverVersion = 'latest';
            this._serverReferenceKey = key;
            await load.apply(this, values);
            await this.refreshServerVersions();
        };
        methods.changeServerVersion = async function() { await this.loadRun(); };
        methods.saveOverwrite = async function() {
            if (!this.loadedRunRef || !confirm('Save a NEW server analysis version? The original backup will remain unchanged.')) return;
            this.isSaving = true;
            try {
                await axios.post('/archive/overwrite', this.buildArchivePayload());
                this.serverVersion = 'latest';
                await this.loadRun();
            } catch (error) { alert(error.response?.data?.detail || error.message); }
            finally { this.isSaving = false; }
        };
        // These actions concern acquisition/controller state, not archive analysis.
        for (const name of ['applyOptimizedBetaToSettings', 'prepareMidFringeSchedule', 'retrySyncArchive', 'retryArchivePhaseAnalysisSync', 'openPreparedMidFringeQueue']) {
            if (methods[name]) methods[name] = function() { alert('Controller actions are disabled on Archive Server. No data is written back.'); };
        }
        for (const name of ['openFolderDialog', 'openFavoriteEditDialog', 'deleteCollectionFolder', 'removeCollectionFavorite', 'runCollectionBatch']) {
            const original = methods[name];
            methods[name] = function(...values) {
                const source = name === 'openFolderDialog' ? Number(values[1] || 0) < 0
                    : name === 'openFavoriteEditDialog' || name === 'removeCollectionFavorite'
                    ? values[0]?.id < 0 : name === 'runCollectionBatch'
                    ? this.selectedCollectionFavoriteIds.some(id => id < 0)
                    : this.selectedCollectionFolderId < 0;
                if (source) { alert('Source Collections are read-only. Create a server Collection to organize your own results.'); return; }
                return original.apply(this, values);
            };
        }
        const folderOptions = methods.collectionFolderOptions;
        methods.collectionFolderOptions = function(...values) { return folderOptions.apply(this, values).filter(folder => folder.id > 0); };
        for (const name of ['openFavoriteDialogForTimelineRun', 'openFavoriteDialogForLoadedRun']) {
            const original = methods[name];
            methods[name] = function(...values) {
                const own = this.collectionFolders.filter(folder => folder.id > 0);
                if (!own.length) { this.archiveSourceMode = 'collections'; this.openFolderDialog('create'); return; }
                original.apply(this, values);
                if (this.collectionDialog.folder_id < 0) this.collectionDialog.folder_id = own[0].id;
            };
        }
        const mounted = options.mounted;
        options.mounted = function() {
            app = this;
            document.title = `${device} — MIGA Archive`;
            axios.get('/archive/device').then(response => { this.serverDeviceName = response.data.name; });
            mounted.call(this);
        };
        return create.call(this, options, ...args);
    };
})();
