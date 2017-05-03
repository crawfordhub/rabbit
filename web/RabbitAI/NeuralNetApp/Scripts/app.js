var ViewModel = function () {
    var self = this;
    self.neuralNetworks = ko.observableArray();
    self.error = ko.observable();

    var neuralNetworksUri = '/api/neuralNetworks/';

    function ajaxHelper(uri, method, data) {
        self.error(''); // Clear error message
        return $.ajax({
            type: method,
            url: uri,
            dataType: 'json',
            contentType: 'application/json',
            data: data ? JSON.stringify(data) : null
        }).fail(function (jqXHR, textStatus, errorThrown) {
            self.error(errorThrown);
        });
    }

    function getAllNeuralNetworks() {
        ajaxHelper(neuralNetworksUri, 'GET').done(function (data) {
            self.neuralNetworks(data);
        });
    }

    // Fetch the initial data.
    getAllNeuralNetworks();
};

ko.applyBindings(new ViewModel());